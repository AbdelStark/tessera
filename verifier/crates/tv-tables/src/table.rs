// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A table regenerated from its kind: the parameter checks, the generators, the digest, the rows
//! and the structured evaluation (RFC-0008 'Table specifications', 'Generators and digests',
//! 'Fixed-point generators', 'Structured evaluation').

use core::fmt;

use tv_codec::Rational;
use tv_field::{Fp, Fp2};
use tv_hash::{Digest, Domain, DomainHasher};

use crate::TableError;
use crate::evaluation::{dense, eq_at, prefix};
use crate::fixed::{LN2, exp_neg, exp_pos};
use crate::galois::{multiply, sbox};
use crate::rope::{Rope, RopeParameters};
use crate::spec::{BitOp, TableId, TableKind, TableSpec, encode_head, length};

/// The most values a dense core may hold (`Exp` and `Silu`), and the most pairs of a `Rope`
/// table: `2^24` (RFC-0008 'Table specifications').
pub const MAX_CORE: u64 = 1 << 24;

/// `Exp`'s first tail rule: a key with `⌊k · scale⌋ ≥ 2^20` has value 0.
const EXP_QUOTIENT_BOUND: u128 = 1 << 20;

/// 1 in Q64.
const ONE_Q64: u128 = 1 << 64;

/// How many bytes of the digest's input reach the hasher at once.
const HASH_BUFFER: usize = 1 << 16;

/// A table regenerated from its kind and identifier, ready for its digest to be compared and its
/// extension evaluated.
///
/// It holds what defines its rows, never the rows themselves: `Range` and `Requant` are closed
/// forms of their parameters, `Exp` and `Silu` hold their dense cores, `Pow2`, `Bitwise8`,
/// `AesSbox`, `Gf256Mul` and `Exp2Frac` every value, and `Rope` its frequencies.
pub struct Table {
    spec: TableSpec,
    rows: u64,
    body: Body,
}

/// What a table holds, by kind.
enum Body {
    Range,
    Requant(Requant),
    /// The value of every row: `Pow2`, `Bitwise8`, `AesSbox`, `Gf256Mul` and `Exp2Frac`.
    Listed(Vec<i64>),
    /// The values of keys `0` to `z − 1`; every key from `z` on has value 0. They lie in
    /// `[0, 2^σ]`, and key 0's is exactly `2^σ`, `2^32` at `σ = 32`: one more than a `u32`
    /// holds, so they are kept as `i64`, 128 MiB at the limit of `2^24` values.
    Exp(Vec<i64>),
    Silu(Silu),
    Rope(Rope),
}

/// A `Requant` table's parameters, checked.
struct Requant {
    b: u8,
    shifts: Vec<u8>,
    lo: i32,
    hi: i32,
}

/// A `Silu` table's dense core: the value of every `h` of every class.
struct Silu {
    b: u8,
    classes: Vec<SiluClass>,
}

/// One class of a `Silu` table: its shift `r`, and its values at `h` from `−2^{b−1−r}` to
/// `2^{b−1−r} − 1`, ascending.
struct SiluClass {
    shift: u8,
    values: Vec<i8>,
}

/// What the parameter checks derive: the number of rows `T`, and an `Exp` table's `z`.
struct Shape {
    rows: u64,
    tail: u64,
}

impl Shape {
    fn rows(rows: u64) -> Self {
        Self { rows, tail: 0 }
    }
}

/// A table regenerated from a borrowed kind, before the kind is copied into it.
struct Regenerated {
    rows: u64,
    log2_rows: u8,
    body: Body,
    digest: Digest,
}

impl Regenerated {
    /// Checks `kind`'s parameters, then a claimed `log2_rows` against `⌈log2 T⌉`, reading both
    /// in place, and only then generates the rows' defining data and the digest.
    fn new(id: TableId, kind: &TableKind, claimed: Option<u8>) -> Result<Self, TableError> {
        let shape = check(kind)?;
        let log2_rows = ceil_log2(shape.rows)?;
        if let Some(found) = claimed
            && found != log2_rows
        {
            return Err(TableError::Log2Rows {
                expected: log2_rows,
                found,
            });
        }
        let body = Body::generate(kind, &shape)?;
        let digest = body.digest(id, kind, log2_rows)?;
        Ok(Self {
            rows: shape.rows,
            log2_rows,
            body,
            digest,
        })
    }

    /// The table of `id` and the `kind` this was regenerated from.
    fn into_table(self, id: TableId, kind: TableKind) -> Table {
        Table {
            spec: TableSpec {
                id,
                kind,
                log2_rows: self.log2_rows,
                digest: self.digest,
            },
            rows: self.rows,
            body: self.body,
        }
    }
}

impl Table {
    /// The table `kind` defines under `id`, its parameters checked in the order of the crate
    /// documentation ('Parameters'): its rows' defining data and its digest, with `log2_rows`
    /// `⌈log2 T⌉`. [`spec`](Self::spec) is the specification a claim list must state for it.
    pub fn generate(id: TableId, kind: TableKind) -> Result<Self, TableError> {
        Ok(Regenerated::new(id, &kind, None)?.into_table(id, kind))
    }

    /// Claim-list validation of one specification (RFC-0008 'Table specifications', RFC-0007
    /// 'Verifier validation' step 4): the table is regenerated from `spec.id` and `spec.kind`,
    /// and the first failure in this order is the error:
    ///
    /// 1. the kind's parameters, each rule in turn;
    /// 2. `spec.log2_rows = ⌈log2 T⌉`, else [`TableError::Log2Rows`];
    /// 3. `spec.digest` is the regenerated digest, else [`TableError::Digest`].
    ///
    /// Each is `Reject(StatementMismatch)`. The first two read `spec` where it lies, and nothing
    /// is copied, allocated or generated before they pass, so a specification they refuse costs
    /// no more than its checks, however many classes it holds: their number is the first thing
    /// read of them. The third generates the table within the limits of the crate documentation
    /// ('Parameters'), and only a specification that passes all three, with at most four
    /// classes, is copied into the table.
    pub fn from_spec(spec: &TableSpec) -> Result<Self, TableError> {
        let regenerated = Regenerated::new(spec.id, &spec.kind, Some(spec.log2_rows))?;
        if regenerated.digest != spec.digest {
            return Err(TableError::Digest);
        }
        Ok(regenerated.into_table(spec.id, spec.kind.clone()))
    }

    /// The regenerated specification: the identifier and kind given, `⌈log2 T⌉` and the digest.
    pub fn spec(&self) -> &TableSpec {
        &self.spec
    }

    /// `T`, the number of rows before padding; for `Rope`, its pairs.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// `⌈log2 T⌉`: the table has `2^{log2_rows}` padded rows, and its extension that many
    /// variables.
    pub fn log2_rows(&self) -> u8 {
        self.spec.log2_rows
    }

    /// The digest, `H("table", TCE(spec without digest) ‖ TCE(pieces) ‖ TCE(core))`.
    pub fn digest(&self) -> &Digest {
        &self.spec.digest
    }

    /// The piece parameters the digest hashes: `[z]` for `Exp`, the first key of its zero tail,
    /// and none for the other kinds.
    pub fn pieces(&self) -> Vec<u64> {
        self.body.pieces()
    }

    /// The true integer value of row `index < T`, or `None` past the rows and for `Rope`, whose
    /// rows hold pairs ([`rope`](Self::rope)). A `Range` row's value is 0.
    pub fn value(&self, index: u64) -> Option<i64> {
        if index >= self.rows {
            return None;
        }
        match &self.body {
            Body::Range => Some(0),
            Body::Requant(requant) => requant.value(index),
            Body::Listed(values) => values.get(usize::try_from(index).ok()?).copied(),
            Body::Exp(core) => Some(core.get(usize::try_from(index).ok()?).map_or(0, |&v| v)),
            Body::Silu(silu) => silu.value(index),
            Body::Rope(_) => None,
        }
    }

    /// Row `index` of the `2^{log2_rows}` padded rows, `(key, value)` in `Fp`: rows from `T` on
    /// are copies of row 0. `None` past the padded rows, and for `Rope`, which is not a lookup
    /// table.
    pub fn row(&self, index: u64) -> Option<(Fp, Fp)> {
        if matches!(self.body, Body::Rope(_)) {
            return None;
        }
        let padded = 1u64.checked_shl(u32::from(self.spec.log2_rows))?;
        if index >= padded {
            return None;
        }
        let row = if index < self.rows { index } else { 0 };
        Some((Fp::reduce(row), field(self.value(row)?)))
    }

    /// A `Rope` table's constants, `None` for the other kinds.
    pub fn rope(&self) -> Option<&Rope> {
        match &self.body {
            Body::Rope(rope) => Some(rope),
            _ => None,
        }
    }

    /// `t̃(ρ) = kẽy(ρ) + β·valũe(ρ)`: the extension of `key + β·value` over the `2^{log2_rows}`
    /// padded rows, whose rows from `T` on are copies of row 0 (RFC-0008 'Structured
    /// evaluation'). `ρ` lists `log2_rows` coordinates, the lowest variable first, else
    /// `PointLength`; a `Rope` table has no extension, `NotLookupTable`. Both are the caller's
    /// errors: a claim list that looks up a `Rope` table must be refused with
    /// `Reject(StatementMismatch)` at its validation (RFC-0007 'Verifier validation' step 5)
    /// before this is called.
    ///
    /// `t̃(ρ) = Σ_{j<T} eq(ρ, j)·t_j + t_0·(1 − LT(ρ, T))`. Every key is `j` and row 0's is 0, so
    /// `kẽy(ρ)` is the prefix sum of `j = Σ_i 2^i·j_i` below `T`. `valũe(ρ)` is
    /// `Σ_{j<T} eq(ρ, j)·value_j + value_0·(1 − LT(ρ, T))`, the sum by kind: 0 for `Range`;
    /// closed forms of `LT` and affine prefix sums per class for `Requant`; the dense core
    /// folded per class over its high variables for `Silu`, and over all for `Exp` and the
    /// listed kinds.
    pub fn eval_mle(&self, rho: &[Fp2], beta: Fp2) -> Result<Fp2, TableError> {
        if matches!(self.body, Body::Rope(_)) {
            return Err(TableError::NotLookupTable);
        }
        if rho.len() != usize::from(self.spec.log2_rows) {
            return Err(TableError::PointLength {
                expected: self.spec.log2_rows,
                found: rho.len(),
            });
        }
        let powers = powers_of_two(rho.len());
        let (lt, key) = prefix(rho, self.rows, Fp::ZERO, &powers).ok_or(TableError::Internal)?;
        let first = field(self.value(0).ok_or(TableError::Internal)?);
        let value = self.body.extension(rho)? + (Fp2::ONE - lt) * first;
        Ok(key + beta * value)
    }
}

/// Shows the specification and the row count, never the cores, which may hold millions of
/// values.
impl fmt::Debug for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Table")
            .field("spec", &self.spec)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

// ----- Parameters -------------------------------------------------------------------------------

/// The parameter checks of RFC-0008 'Table specifications', each kind's rules in the order the
/// crate documentation lists them, the first that fails the error.
fn check(kind: &TableKind) -> Result<Shape, TableError> {
    match kind {
        TableKind::Range { bits } => {
            if !(1..=32).contains(bits) {
                return Err(TableError::RangeBits { bits: *bits });
            }
            Ok(Shape::rows(power_of_two(*bits)?))
        }
        TableKind::Requant {
            b,
            r_by_class,
            lo,
            hi,
        } => {
            let rows = class_rows(*b, r_by_class)?;
            for (class, &shift) in r_by_class.iter().enumerate() {
                if !(1..=63).contains(&shift) {
                    return Err(TableError::RequantShift { class, shift });
                }
            }
            if lo > hi {
                return Err(TableError::RequantClamp { lo: *lo, hi: *hi });
            }
            Ok(Shape::rows(rows))
        }
        TableKind::Pow2 { max_exp } => {
            if !(1..=63).contains(max_exp) {
                return Err(TableError::Pow2Exponent { max_exp: *max_exp });
            }
            Ok(Shape::rows(u64::from(*max_exp)))
        }
        TableKind::Bitwise8 { .. } => Ok(Shape::rows(1 << 16)),
        TableKind::AesSbox | TableKind::Gf256Mul { .. } => Ok(Shape::rows(1 << 8)),
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        } => {
            if !(1..=32).contains(key_bits) {
                return Err(TableError::ExpKeyBits {
                    key_bits: *key_bits,
                });
            }
            if *sigma > 32 {
                return Err(TableError::ExpSigma { sigma: *sigma });
            }
            if scale.num() <= 0 {
                return Err(TableError::ExpScale);
            }
            let rows = power_of_two(*key_bits)?;
            let tail = zero_tail(rows, *sigma, *scale)?;
            if tail > MAX_CORE {
                return Err(TableError::ExpTail { tail });
            }
            Ok(Shape { rows, tail })
        }
        TableKind::Exp2Frac {
            frac_bits,
            out_bits,
        } => {
            if !(1..=16).contains(frac_bits) {
                return Err(TableError::Exp2FracBits {
                    frac_bits: *frac_bits,
                });
            }
            if !(1..=56).contains(out_bits) {
                return Err(TableError::Exp2FracOutBits {
                    out_bits: *out_bits,
                });
            }
            Ok(Shape::rows(power_of_two(*frac_bits)?))
        }
        TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        } => {
            let rows = class_rows(*b, r_by_class)?;
            let mut values: u64 = 0;
            for (class, &shift) in r_by_class.iter().enumerate() {
                if shift >= *b {
                    return Err(TableError::SiluShift { class, shift });
                }
            }
            for &shift in r_by_class {
                let size = power_of_two(b.checked_sub(shift).ok_or(TableError::Internal)?)?;
                values = values.checked_add(size).ok_or(TableError::Internal)?;
            }
            if values > MAX_CORE {
                return Err(TableError::SiluCore { values });
            }
            if !(-64..=0).contains(in_exp) {
                return Err(TableError::SiluInExp { in_exp: *in_exp });
            }
            let shift = silu_shift(*in_exp, *out_exp);
            if !(1..=126).contains(&shift) {
                return Err(TableError::SiluOutShift { shift });
            }
            Ok(Shape::rows(rows))
        }
        TableKind::Rope {
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max,
            bits,
            max_context,
            d_head,
        } => {
            if *theta < 2 {
                return Err(TableError::RopeTheta { theta: *theta });
            }
            if *factor == 0 {
                return Err(TableError::RopeFactor);
            }
            if *low_freq_factor == 0 || low_freq_factor >= high_freq_factor {
                return Err(TableError::RopeFrequencyFactors {
                    low: *low_freq_factor,
                    high: *high_freq_factor,
                });
            }
            if *original_max == 0 {
                return Err(TableError::RopeOriginalMax);
            }
            if !(1..=30).contains(bits) {
                return Err(TableError::RopeBits { bits: *bits });
            }
            if *d_head == 0 || !d_head.is_multiple_of(2) {
                return Err(TableError::RopeHeadDim { d_head: *d_head });
            }
            if *max_context == 0 {
                return Err(TableError::RopeContext);
            }
            let half = u64::from(d_head.checked_div(2).ok_or(TableError::Internal)?);
            let pairs = u64::from(*max_context)
                .checked_mul(half)
                .ok_or(TableError::Internal)?;
            if pairs > MAX_CORE {
                return Err(TableError::RopePairs { pairs });
            }
            Ok(Shape::rows(pairs))
        }
    }
}

/// The rules `Requant` and `Silu` share, in order: `b ≥ 1`, `1 ≤ n ≤ 4` and
/// `b + ⌈log2 n⌉ ≤ 32`. Returns `T = n · 2^b`.
fn class_rows(b: u8, shifts: &[u8]) -> Result<u64, TableError> {
    if b == 0 {
        return Err(TableError::KeyBits);
    }
    let classes = shifts.len();
    if !(1..=4).contains(&classes) {
        return Err(TableError::Classes { classes });
    }
    let class_bits = ceil_log2(length(classes))?;
    if u32::from(b).saturating_add(u32::from(class_bits)) > 32 {
        return Err(TableError::RowBits { b, classes });
    }
    length(classes)
        .checked_mul(power_of_two(b)?)
        .ok_or(TableError::Internal)
}

/// `2^bits` for `bits < 64`.
fn power_of_two(bits: u8) -> Result<u64, TableError> {
    1u64.checked_shl(u32::from(bits))
        .ok_or(TableError::Internal)
}

/// `⌈log2 rows⌉` for `rows ≥ 1`: 0 for one row.
fn ceil_log2(rows: u64) -> Result<u8, TableError> {
    let below = rows.checked_sub(1).ok_or(TableError::Internal)?;
    let bits = u64::BITS
        .checked_sub(below.leading_zeros())
        .ok_or(TableError::Internal)?;
    u8::try_from(bits).map_err(|_| TableError::Internal)
}

/// `Silu`'s output shift `s = 63 − in_exp + out_exp`.
fn silu_shift(in_exp: i8, out_exp: i8) -> i16 {
    // Both exponents fit an `i8`, so the result lies in −192..=318 and fits an `i16`.
    63i16
        .saturating_sub(i16::from(in_exp))
        .saturating_add(i16::from(out_exp))
}

// ----- Generators -------------------------------------------------------------------------------

impl Body {
    /// Generates what a checked kind's rows are defined by.
    fn generate(kind: &TableKind, shape: &Shape) -> Result<Self, TableError> {
        Ok(match kind {
            TableKind::Range { .. } => Body::Range,
            TableKind::Requant {
                b,
                r_by_class,
                lo,
                hi,
            } => Body::Requant(Requant {
                b: *b,
                shifts: r_by_class.clone(),
                lo: *lo,
                hi: *hi,
            }),
            TableKind::Pow2 { .. } => Body::Listed(listed(shape.rows, |j| {
                let power = 1u64
                    .checked_shl(u32::try_from(j).map_err(|_| TableError::Internal)?)
                    .ok_or(TableError::Internal)?;
                i64::try_from(power).map_err(|_| TableError::Internal)
            })?),
            TableKind::Bitwise8 { op } => Body::Listed(listed(shape.rows, |j| {
                let (a, b) = (j & 0xFF, j >> 8);
                let value = match op {
                    BitOp::Xor => a ^ b,
                    BitOp::And => a & b,
                };
                i64::try_from(value).map_err(|_| TableError::Internal)
            })?),
            TableKind::AesSbox => {
                Body::Listed(listed(shape.rows, |j| Ok(i64::from(sbox(byte(j)?))))?)
            }
            TableKind::Gf256Mul { c } => Body::Listed(listed(shape.rows, |j| {
                Ok(i64::from(multiply(*c, byte(j)?)))
            })?),
            TableKind::Exp { sigma, scale, .. } => {
                Body::Exp(listed(shape.tail, |k| exp_value(k, *sigma, *scale))?)
            }
            TableKind::Exp2Frac {
                frac_bits,
                out_bits,
            } => Body::Listed(listed(shape.rows, |f| {
                exp2frac_value(f, *frac_bits, *out_bits)
            })?),
            TableKind::Silu {
                b,
                r_by_class,
                in_exp,
                out_exp,
            } => Body::Silu(Silu::generate(*b, r_by_class, *in_exp, *out_exp)?),
            TableKind::Rope {
                theta,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max,
                bits,
                max_context,
                d_head,
            } => Body::Rope(Rope::new(&RopeParameters {
                theta: *theta,
                factor: *factor,
                low_freq_factor: *low_freq_factor,
                high_freq_factor: *high_freq_factor,
                original_max: *original_max,
                bits: *bits,
                max_context: *max_context,
                d_head: *d_head,
            })?),
        })
    }
}

/// The values of keys `0` to `count − 1`, each from `value`, with room for all of them reserved
/// first: memory that cannot be reserved is `Memory`.
fn listed(
    count: u64,
    value: impl Fn(u64) -> Result<i64, TableError>,
) -> Result<Vec<i64>, TableError> {
    let len = usize::try_from(count).map_err(|_| TableError::Memory)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| TableError::Memory)?;
    for key in 0..count {
        values.push(value(key)?);
    }
    Ok(values)
}

/// A key of a table of `2^8` rows as a byte.
fn byte(key: u64) -> Result<u8, TableError> {
    u8::try_from(key).map_err(|_| TableError::Internal)
}

/// An `Exp` key's argument.
enum ExpArgument {
    /// `⌊k · scale⌋ ≥ 2^20`: the key is in the zero tail by the first rule.
    Tail,
    /// `x = q · 2^64 + ⌊m · 2^64 / den⌋`, the floor of `k · scale · 2^64`, where
    /// `k · num = q · den + m`.
    Point(u128),
}

/// Steps 1 to 3 of RFC-0008's `Exp`: `k · num = q · den + m`; the tail when `q ≥ 2^20`; else
/// `x = q · 2^64 + ⌊m · 2^64 / den⌋`. `k < 2^32` and `num < 2^63` keep `k · num` below `2^95`,
/// and `m < den` keeps `m · 2^64` below `2^128`.
fn exp_argument(key: u64, scale: Rational) -> Result<ExpArgument, TableError> {
    let internal = TableError::Internal;
    let num = u128::try_from(scale.num()).map_err(|_| internal)?;
    let den = u128::from(scale.den());
    let product = u128::from(key).checked_mul(num).ok_or(internal)?;
    let q = product.checked_div(den).ok_or(internal)?;
    let m = product.checked_rem(den).ok_or(internal)?;
    if q >= EXP_QUOTIENT_BOUND {
        return Ok(ExpArgument::Tail);
    }
    let fraction = m
        .checked_mul(ONE_Q64)
        .and_then(|scaled| scaled.checked_div(den))
        .ok_or(internal)?;
    let x = q
        .checked_mul(ONE_Q64)
        .and_then(|whole| whole.checked_add(fraction))
        .ok_or(internal)?;
    Ok(ExpArgument::Point(x))
}

/// `A = (σ + 2) · LN2`: a key whose `x` reaches it has `n = ⌊x / LN2⌋ ≥ σ + 2`.
fn tail_threshold(sigma: u8) -> Result<u128, TableError> {
    u128::from(sigma)
        .checked_add(2)
        .and_then(|n| n.checked_mul(LN2))
        .ok_or(TableError::Internal)
}

/// Whether key `k` is in `Exp`'s zero tail by RFC-0008's rule, key by key: `q ≥ 2^20`, or
/// `x ≥ (σ + 2) · LN2`.
fn in_exp_tail(key: u64, sigma: u8, scale: Rational) -> Result<bool, TableError> {
    Ok(match exp_argument(key, scale)? {
        ExpArgument::Tail => true,
        ExpArgument::Point(x) => x >= tail_threshold(sigma)?,
    })
}

/// `⌈a / b⌉`, or `None` for `b = 0`.
fn ceil_div(a: u128, b: u128) -> Option<u128> {
    let quotient = a.checked_div(b)?;
    let remainder = a.checked_rem(b)?;
    quotient.checked_add(u128::from(remainder != 0))
}

/// `z`, the first key of `Exp`'s zero tail, or `rows` when no key is in it, from a closed form
/// of the tail's two rules rather than a search over keys. With `scale = num / den`:
///
/// - `q = ⌊k·num / den⌋ ≥ 2^20` exactly when `k ≥ k_1 = ⌈2^20·den / num⌉`;
/// - below `k_1`, `x = ⌊k·num·2^64 / den⌋`, and `x ≥ A` for the integer `A = (σ + 2)·LN2`
///   exactly when `k·num·2^64 ≥ A·den`, that is `k ≥ k_2 = ⌈A·den / (num·2^64)⌉`.
///
/// So `z = min(rows, k_1, k_2)`: the tail is the suffix from `z`. `A·den` reaches `2^134`, so
/// `k_2` is `⌈⌈A·den / 2^64⌉ / num⌉`, the inner ceiling `A_hi·den + ⌈A_lo·den / 2^64⌉` for
/// `A = A_hi·2^64 + A_lo`. The rule itself then confirms the boundary: key `z − 1` is outside
/// the tail and key `z`, when it is a key, inside.
fn zero_tail(rows: u64, sigma: u8, scale: Rational) -> Result<u64, TableError> {
    let internal = TableError::Internal;
    let num = u128::try_from(scale.num()).map_err(|_| internal)?;
    let den = u128::from(scale.den());
    let k1 = EXP_QUOTIENT_BOUND
        .checked_mul(den)
        .and_then(|bound| ceil_div(bound, num))
        .ok_or(internal)?;
    let threshold = tail_threshold(sigma)?;
    let (high, low) = (threshold >> 64, threshold & u128::from(u64::MAX));
    let scaled = high
        .checked_mul(den)
        .zip(low.checked_mul(den).and_then(|low| ceil_div(low, ONE_Q64)))
        .and_then(|(high, low)| high.checked_add(low))
        .ok_or(internal)?;
    let k2 = ceil_div(scaled, num).ok_or(internal)?;
    let z = u64::try_from(k1.min(k2).min(u128::from(rows))).map_err(|_| internal)?;
    let below = match z.checked_sub(1) {
        Some(last) => !in_exp_tail(last, sigma, scale)?,
        None => true,
    };
    let at = z >= rows || in_exp_tail(z, sigma, scale)?;
    if below && at { Ok(z) } else { Err(internal) }
}

/// Step 4 of RFC-0008's `Exp`: `(y, n) = exp_neg(x)`; 0 when `n ≥ σ + 2`, else
/// `(y · 2^σ + 2^{63+n}) >> (64 + n)`, rounded half up. `y ≤ 2^64` and `n ≤ σ + 1 ≤ 33` keep
/// the sum below `2^98`.
fn exp_value(key: u64, sigma: u8, scale: Rational) -> Result<i64, TableError> {
    let internal = TableError::Internal;
    let ExpArgument::Point(x) = exp_argument(key, scale)? else {
        return Ok(0);
    };
    let (y, n) = exp_neg(x).ok_or(internal)?;
    if n >= u128::from(sigma).saturating_add(2) {
        return Ok(0);
    }
    let n = u32::try_from(n).map_err(|_| internal)?;
    let half = 1u128.checked_shl(n.saturating_add(63)).ok_or(internal)?;
    let value = y
        .checked_mul(1u128.checked_shl(u32::from(sigma)).ok_or(internal)?)
        .and_then(|scaled| scaled.checked_add(half))
        .and_then(|sum| sum.checked_shr(n.saturating_add(64)))
        .ok_or(internal)?;
    i64::try_from(value).map_err(|_| internal)
}

/// `Exp2Frac` of key `f`: `exp_pos(⌊f · LN2 / 2^frac_bits⌋) >> (62 − out_bits)`. The argument
/// is below `LN2`, as `exp_pos` needs.
fn exp2frac_value(f: u64, frac_bits: u8, out_bits: u8) -> Result<i64, TableError> {
    let internal = TableError::Internal;
    let argument = u128::from(f)
        .checked_mul(LN2)
        .and_then(|product| product.checked_shr(u32::from(frac_bits)))
        .ok_or(internal)?;
    let y = exp_pos(argument).ok_or(internal)?;
    let shift = 62u32.checked_sub(u32::from(out_bits)).ok_or(internal)?;
    i64::try_from(y.checked_shr(shift).ok_or(internal)?).map_err(|_| internal)
}

/// `Silu` at `h`, `|h| < 2^31`:
///
/// 1. `(y, n) = exp_neg(|h| · 2^{64 + in_exp})` and `E = y >> n`, or 0 when `n ≥ 128`:
///    `e^{−|x|}` in Q64 for `x = h · 2^{in_exp}`;
/// 2. `σ = ⌊2^127 / (2^64 + E)⌋` when `h ≥ 0` and `⌊E · 2^63 / (2^64 + E)⌋` when `h < 0`, the
///    logistic `σ(x)` in Q63;
/// 3. `clamp((h · σ + 2^{s−1}) >> s, −128, 127)` with `s = 63 − in_exp + out_exp`.
///
/// The argument stays below `2^95`, and `|h · σ| < 2^94`.
fn silu_value(h: i64, in_exp: i8, out_exp: i8) -> Result<i8, TableError> {
    let internal = TableError::Internal;
    let lift = u32::try_from(64i32.saturating_add(i32::from(in_exp))).map_err(|_| internal)?;
    let argument = u128::from(h.unsigned_abs())
        .checked_mul(1u128.checked_shl(lift).ok_or(internal)?)
        .ok_or(internal)?;
    let (y, n) = exp_neg(argument).ok_or(internal)?;
    let e = match u32::try_from(n) {
        Ok(n) if n < u128::BITS => y.checked_shr(n).ok_or(internal)?,
        _ => 0,
    };
    let denominator = ONE_Q64.checked_add(e).ok_or(internal)?;
    let numerator = if h >= 0 {
        1u128 << 127
    } else {
        e.checked_mul(1 << 63).ok_or(internal)?
    };
    let logistic = numerator.checked_div(denominator).ok_or(internal)?;
    let shift = u32::try_from(silu_shift(in_exp, out_exp)).map_err(|_| internal)?;
    let half = 1i128
        .checked_shl(shift.checked_sub(1).ok_or(internal)?)
        .ok_or(internal)?;
    let rounded = i128::from(h)
        .checked_mul(i128::try_from(logistic).map_err(|_| internal)?)
        .and_then(|product| product.checked_add(half))
        .and_then(|sum| sum.checked_shr(shift))
        .ok_or(internal)?;
    i8::try_from(rounded.clamp(-128, 127)).map_err(|_| internal)
}

impl Silu {
    /// The dense core of checked parameters: class by class, `h` ascending.
    fn generate(b: u8, shifts: &[u8], in_exp: i8, out_exp: i8) -> Result<Self, TableError> {
        let internal = TableError::Internal;
        let mut classes = Vec::new();
        classes
            .try_reserve_exact(shifts.len())
            .map_err(|_| TableError::Memory)?;
        for &shift in shifts {
            let bits = b.checked_sub(shift).ok_or(internal)?;
            let count = power_of_two(bits)?;
            // The class's first `h`, `−2^{b−1−r}`.
            let first = power_of_two(bits.checked_sub(1).ok_or(internal)?)
                .ok()
                .and_then(|half| i64::try_from(half).ok())
                .map(i64::wrapping_neg)
                .ok_or(internal)?;
            let len = usize::try_from(count).map_err(|_| TableError::Memory)?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(len)
                .map_err(|_| TableError::Memory)?;
            for u in 0..count {
                let h = i64::try_from(u)
                    .ok()
                    .and_then(|u| first.checked_add(u))
                    .ok_or(internal)?;
                values.push(silu_value(h, in_exp, out_exp)?);
            }
            classes.push(SiluClass { shift, values });
        }
        Ok(Self { b, classes })
    }

    /// The value of row `index`: class `⌊index / 2^b⌋`, and its core's entry at
    /// `(index mod 2^b) >> r`, which is `h + 2^{b−1−r}`.
    fn value(&self, index: u64) -> Option<i64> {
        let class = usize::try_from(index.checked_shr(u32::from(self.b))?).ok()?;
        let entry = self.classes.get(class)?;
        let x = index & low_mask(self.b)?;
        let at = usize::try_from(x.checked_shr(u32::from(entry.shift))?).ok()?;
        entry.values.get(at).map(|&value| i64::from(value))
    }

    /// `Σ_{j<T} eq(ρ, j)·value_j`: class `c`'s rows depend on the high `b − r_c` of their low `b`
    /// bits only, so its sum is its core's extension at `ρ_{r_c}, …, ρ_{b−1}`, the variables
    /// below `r_c` summing out, times `eq` of the class bits.
    fn extension(&self, rho: &[Fp2]) -> Result<Fp2, TableError> {
        let (low, high) = rho
            .split_at_checked(usize::from(self.b))
            .ok_or(TableError::Internal)?;
        let mut sum = Fp2::ZERO;
        for (class, entry) in self.classes.iter().enumerate() {
            let vars = low
                .get(usize::from(entry.shift)..)
                .ok_or(TableError::Internal)?;
            let core = dense(&entry.values, vars, |value| field(i64::from(value)))?;
            sum += eq_at(high, length(class)) * core;
        }
        Ok(sum)
    }
}

/// `2^b − 1`, the bits of a key within its class.
fn low_mask(b: u8) -> Option<u64> {
    1u64.checked_shl(u32::from(b))?.checked_sub(1)
}

impl Requant {
    /// The value of row `index`: `rq(k, r_c, lo, hi)` for class `c = ⌊index / 2^b⌋` and
    /// `k = (index mod 2^b) − 2^{b−1}`.
    fn value(&self, index: u64) -> Option<i64> {
        let class = usize::try_from(index.checked_shr(u32::from(self.b))?).ok()?;
        let shift = *self.shifts.get(class)?;
        let x = i64::try_from(index & low_mask(self.b)?).ok()?;
        let offset = i64::try_from(1u64.checked_shl(u32::from(self.b).checked_sub(1)?)?).ok()?;
        rq(x.checked_sub(offset)?, shift, self.lo, self.hi)
    }

    /// `Σ_{j<T} eq(ρ, j)·value_j`, class by class: the class's sum over its low `b` variables,
    /// times `eq` of the class bits.
    fn extension(&self, rho: &[Fp2]) -> Result<Fp2, TableError> {
        let (low, high) = rho
            .split_at_checked(usize::from(self.b))
            .ok_or(TableError::Internal)?;
        let mut sum = Fp2::ZERO;
        for (class, &shift) in self.shifts.iter().enumerate() {
            let local = requant_class(low, self.b, shift, self.lo, self.hi)?;
            sum += eq_at(high, length(class)) * local;
        }
        Ok(sum)
    }
}

/// `rq(k, r, lo, hi) = clamp(⌊(k + 2^{r−1}) / 2^r⌋, lo, hi)` for `r ≥ 1` (RFC-0005 'Integer
/// semantics'): round half up, saturate. `>>` on `i128` is floor division by `2^r`.
fn rq(k: i64, shift: u8, lo: i32, hi: i32) -> Option<i64> {
    let half = 1i128.checked_shl(u32::from(shift).checked_sub(1)?)?;
    let floor = i128::from(k)
        .checked_add(half)?
        .checked_shr(u32::from(shift))?;
    i64::try_from(floor.clamp(i128::from(lo), i128::from(hi))).ok()
}

/// One `Requant` class over its `b` low variables, in `O(b)`. With `x = k + 2^{b−1}` the offset
/// key, the unclamped value `f(x) = ⌊(x − 2^{b−1} + 2^{r−1}) / 2^r⌋` is affine in the bits of
/// `x` ([`rounding_form`]) and never decreases, so the class is `lo` on `[0, start)`, `f` on
/// `[start, end)` and `hi` on `[end, 2^b)` ([`clamp_bounds`]), and its sum is
/// `lo·LT(ρ, start) + (S(ρ, end) − S(ρ, start)) + hi·(1 − LT(ρ, end))`, `S` the prefix sum of
/// `f`.
fn requant_class(low: &[Fp2], b: u8, shift: u8, lo: i32, hi: i32) -> Result<Fp2, TableError> {
    let internal = TableError::Internal;
    let (start, end) = clamp_bounds(b, shift, lo, hi).ok_or(internal)?;
    let (w0, weights) = rounding_form(b, shift).ok_or(internal)?;
    let (lt_start, sum_start) = prefix(low, start, w0, &weights).ok_or(internal)?;
    let (lt_end, sum_end) = prefix(low, end, w0, &weights).ok_or(internal)?;
    let lo = field(i64::from(lo));
    let hi = field(i64::from(hi));
    Ok(lt_start * lo + (sum_end - sum_start) + (Fp2::ONE - lt_end) * hi)
}

/// The affine form of `f(x) = ⌊(x − 2^{b−1} + 2^{r−1}) / 2^r⌋` in the `b` bits `x_i` of `x`:
/// for `r < b`, `f(x) = (x >> r) + x_{r−1} − 2^{b−1−r}` (the rounding reads bit `r − 1`), so
/// `w_0 = −2^{b−1−r}`, `w_{r−1} = 1` and `w_i = 2^{i−r}` for `i ≥ r`; for `r ≥ b`, `f` is 0.
fn rounding_form(b: u8, shift: u8) -> Option<(Fp, Vec<Fp>)> {
    let mut weights = vec![Fp::ZERO; usize::from(b)];
    if shift >= b {
        return Some((Fp::ZERO, weights));
    }
    let r = usize::from(shift);
    for (i, weight) in weights.iter_mut().enumerate() {
        if let Some(above) = i.checked_sub(r) {
            *weight = Fp::reduce(1u64.checked_shl(u32::try_from(above).ok()?)?);
        } else if i.checked_add(1) == Some(r) {
            *weight = Fp::ONE;
        }
    }
    let constant = 1u64.checked_shl(u32::from(b.checked_sub(1)?.checked_sub(shift)?))?;
    Some((-Fp::reduce(constant), weights))
}

/// `start`, the number of offset keys `x ∈ [0, 2^b)` with `f(x) ≤ lo`, and `end`, the number
/// with `f(x) < hi`, but at least `start`. `f(x) ≤ v` exactly when
/// `x < (v + 1)·2^r + 2^{b−1} − 2^{r−1}`, so both are such bounds clamped to `[0, 2^b]`. When
/// `lo = hi`, `end = start` and the middle is empty: every value is `lo`.
fn clamp_bounds(b: u8, shift: u8, lo: i32, hi: i32) -> Option<(u64, u64)> {
    let size = 1i128.checked_shl(u32::from(b))?;
    let scale = 1i128.checked_shl(u32::from(shift))?;
    let offset = 1i128
        .checked_shl(u32::from(b.checked_sub(1)?))?
        .checked_sub(1i128.checked_shl(u32::from(shift.checked_sub(1)?))?)?;
    let bound = |level: i128| -> Option<u64> {
        let x = level.checked_mul(scale)?.checked_add(offset)?;
        u64::try_from(x.clamp(0, size)).ok()
    };
    let start = bound(i128::from(lo).checked_add(1)?)?;
    let end = bound(i128::from(hi))?.max(start);
    Some((start, end))
}

impl Body {
    /// `Σ_{j<T} eq(ρ, j)·value_j` over the unpadded rows.
    fn extension(&self, rho: &[Fp2]) -> Result<Fp2, TableError> {
        match self {
            Body::Range => Ok(Fp2::ZERO),
            Body::Requant(requant) => requant.extension(rho),
            Body::Listed(values) | Body::Exp(values) => dense(values, rho, field),
            Body::Silu(silu) => silu.extension(rho),
            Body::Rope(_) => Err(TableError::NotLookupTable),
        }
    }

    /// The piece parameters: `[z]` for `Exp`, none otherwise.
    fn pieces(&self) -> Vec<u64> {
        match self {
            Body::Exp(core) => vec![length(core.len())],
            _ => Vec::new(),
        }
    }

    /// `H("table", TCE(id, kind, log2_rows) ‖ TCE(pieces) ‖ TCE(core))`, the pieces a
    /// `Vec<u64>` and the core a `Vec<Fp>` in row order (RFC-0008 'Generators and digests').
    fn digest(&self, id: TableId, kind: &TableKind, log2_rows: u8) -> Result<Digest, TableError> {
        let mut input = HashInput::new();
        let mut head = Vec::new();
        encode_head(id, kind, log2_rows, &mut head);
        input.push(&head);
        let pieces = self.pieces();
        input.length(pieces.len());
        for piece in &pieces {
            input.push(&piece.to_le_bytes());
        }
        match self {
            Body::Range | Body::Requant(_) => input.length(0),
            Body::Listed(values) | Body::Exp(values) => {
                input.length(values.len());
                for &value in values {
                    input.value(value);
                }
            }
            Body::Silu(silu) => {
                let total = silu
                    .classes
                    .iter()
                    .try_fold(0usize, |total, class| total.checked_add(class.values.len()))
                    .ok_or(TableError::Internal)?;
                input.length(total);
                for class in &silu.classes {
                    for &value in &class.values {
                        input.value(i64::from(value));
                    }
                }
            }
            Body::Rope(rope) => {
                let values = usize::try_from(rope.pairs())
                    .ok()
                    .and_then(|pairs| pairs.checked_mul(2))
                    .ok_or(TableError::Internal)?;
                input.length(values);
                for position in 0..rope.positions() {
                    for frequency in 0..rope.frequencies().len() {
                        let (c, s) = rope.pair(position, frequency).ok_or(TableError::Internal)?;
                        input.value(c);
                        input.value(s);
                    }
                }
            }
        }
        Ok(input.finish())
    }
}

/// The digest's input, handed to the hasher in pieces of [`HASH_BUFFER`] bytes.
struct HashInput {
    hasher: DomainHasher,
    buffer: Vec<u8>,
}

impl HashInput {
    fn new() -> Self {
        Self {
            hasher: DomainHasher::new(Domain::Table),
            buffer: Vec::with_capacity(HASH_BUFFER),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
        if self.buffer.len() >= HASH_BUFFER {
            self.hasher.update(&self.buffer);
            self.buffer.clear();
        }
    }

    /// A sequence's TCE length, a `u64`.
    fn length(&mut self, len: usize) {
        self.push(&length(len).to_le_bytes());
    }

    /// A value as TCE `Fp`, in the centered representation.
    fn value(&mut self, value: i64) {
        self.push(&field(value).to_le_bytes());
    }

    fn finish(mut self) -> Digest {
        self.hasher.update(&self.buffer);
        self.hasher.finalize()
    }
}

/// The true integer `v` in `Fp`, in the centered representation (RFC-0002 'Signed integers'):
/// `v` for `v ≥ 0` and `P − |v|` for `v < 0`. Every table value has `|v| < 2^63 < P`.
pub(crate) fn field(value: i64) -> Fp {
    let magnitude = Fp::reduce(value.unsigned_abs());
    if value < 0 { -magnitude } else { magnitude }
}

/// `2^0, 2^1, …, 2^{n−1}` in `Fp`: the weights of `j = Σ_i 2^i·j_i`.
fn powers_of_two(n: usize) -> Vec<Fp> {
    let mut powers = Vec::with_capacity(n);
    let mut power = Fp::ONE;
    for _ in 0..n {
        powers.push(power);
        power += power;
    }
    powers
}
