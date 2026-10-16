// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tables generated from their kind: the rows the LogUp prover commits to, and the witness-side
//! lookup of a key (RFC-0008 'Interfaces').

use core::hint::select_unpredictable;

use tessera_codec::Encode;
use tessera_field::{Goldilocks, PrimeField};
use tessera_hash::{Digest, Domain, Hasher};

use crate::fixed::{LN2, exp_neg, exp_pos};
use crate::gf256;
use crate::{BitOp, TableError, TableId, TableKind, TableSpec};

/// The longest dense core a table may have (RFC-0008 'Structured evaluation').
const MAX_CORE: u64 = 1 << 24;

/// A lookup table of RFC-0008: `T` rows of `(key, value)`, padded to `2^log2_rows` with copies
/// of row 0.
///
/// [`index`](Table::index) and [`lookup`](Table::lookup) take a witness's key, which is secret
/// (05 §3). The arithmetic kinds compute the row and the value with no memory access that
/// depends on the key, and select a class's shift without indexing by it (INV-ARCH-2). The
/// fixed-point kinds, `Exp` and `Exp2Frac`, read their value from the generated core at the
/// key, an access #257 reconciles with INV-ARCH-2. Whether the key is in the table is the one
/// outcome that depends on it, and an honest witness's keys always are (INV-DM-2).
pub trait Table {
    /// The specification, digest included.
    fn spec(&self) -> &TableSpec;

    /// `⌈log2 T⌉`: the table tree has `2^log2_rows` leaves.
    fn log2_rows(&self) -> u8 {
        self.spec().log2_rows
    }

    /// `T`, the rows before padding.
    fn row_count(&self) -> u64;

    /// Row `index` as `(key, value)` field elements in the centered representation. Rows `T` to
    /// `2^log2_rows − 1` repeat row 0; past them, `None`.
    fn row(&self, index: u64) -> Option<(Goldilocks, Goldilocks)>;

    /// The row that a lookup of `key` in `class` hits, or `None` if the table has no such key.
    /// Tables without classes take class 0.
    fn index(&self, key: i64, class: u8) -> Option<u64>;

    /// The value that a lookup of `key` in `class` returns, or `None` if the table has no such
    /// key.
    fn lookup(&self, key: i64, class: u8) -> Option<i64>;
}

/// A table generated from its kind. The arithmetic kinds compute rows on demand. The
/// fixed-point kinds generate their dense core when built: `Exp` the values below its zero tail
/// (at most `2^24`), `Exp2Frac` all of its at most `2^16`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedTable {
    spec: TableSpec,
    rows: u64,
    /// The values of rows `0` to `core.len() − 1`, for the fixed-point kinds; empty otherwise.
    core: Vec<i64>,
}

impl GeneratedTable {
    /// The table of `kind` with index `id`: its parameters checked against RFC-0008 'Table
    /// specifications', its core generated, and its `log2_rows` and digest computed.
    ///
    /// # Errors
    ///
    /// [`TableError::Parameter`] for parameters the kind does not allow.
    pub fn new(id: TableId, kind: TableKind) -> Result<Self, TableError> {
        let rows = row_count(&kind)?;
        let core = generate_core(&kind, rows)?;
        let mut table = Self {
            spec: TableSpec {
                id,
                log2_rows: ceil_log2(rows),
                kind,
                digest: [0; 32],
            },
            rows,
            core,
        };
        table.spec.digest = table.digest();
        Ok(table)
    }

    /// The table a claim list names, regenerated from its kind and id, as claim-list validation
    /// does (RFC-0007): the specification must be the one its kind generates.
    ///
    /// # Errors
    ///
    /// [`TableError::Parameter`] for parameters the kind does not allow, then
    /// [`TableError::Log2Rows`] and [`TableError::Digest`] for a `log2_rows` or digest other
    /// than the generated one.
    pub fn from_spec(spec: &TableSpec) -> Result<Self, TableError> {
        let table = Self::new(spec.id, spec.kind.clone())?;
        if spec.log2_rows != table.spec.log2_rows {
            return Err(TableError::Log2Rows {
                stated: spec.log2_rows,
                expected: table.spec.log2_rows,
            });
        }
        if spec.digest != table.spec.digest {
            return Err(TableError::Digest);
        }
        Ok(table)
    }

    /// `H("table", TCE(id, kind, log2_rows) ‖ TCE(pieces) ‖ TCE(core))` (RFC-0008 'Generators
    /// and digests'), both sequences of TCE:
    ///
    /// - `Exp` has one piece, the first key of its zero tail, and its core is the values below
    ///   it; `Exp2Frac` has no piece and its core is all of its values;
    /// - the enumerated arithmetic kinds have no piece and their value column as core;
    /// - `Range` and `Requant` have neither: their values are closed forms of their parameters.
    fn digest(&self) -> Digest {
        let mut hasher = Hasher::new(Domain::Table);
        self.spec.id.encode_to(&mut hasher);
        self.spec.kind.encode_to(&mut hasher);
        self.spec.log2_rows.encode_to(&mut hasher);
        let pieces: Vec<u64> = match self.spec.kind {
            TableKind::Exp { .. } => vec![self.core.len() as u64],
            _ => vec![],
        };
        pieces.encode_to(&mut hasher);
        if let TableKind::Silu { .. } = self.spec.kind {
            // The core is every value of `h`, class by class: not the table's rows.
            (self.core.len() as u64).encode_to(&mut hasher);
            for &value in &self.core {
                Goldilocks::from_i64(value).encode_to(&mut hasher);
            }
            return hasher.finalize();
        }
        let core_len = match self.spec.kind {
            TableKind::Range { .. }
            | TableKind::Requant { .. }
            | TableKind::Silu { .. }
            | TableKind::Rope { .. } => 0,
            TableKind::Exp { .. } | TableKind::Exp2Frac { .. } => self.core.len() as u64,
            TableKind::Pow2 { .. }
            | TableKind::Bitwise8 { .. }
            | TableKind::AesSbox
            | TableKind::Gf256Mul { .. } => self.rows,
        };
        core_len.encode_to(&mut hasher);
        for j in 0..core_len {
            Goldilocks::from_i64(self.value(j)).encode_to(&mut hasher);
        }
        hasher.finalize()
    }

    /// The value of row `j < T`, the row whose key is `j`.
    fn value(&self, j: u64) -> i64 {
        // `j < T ≤ 2^32`, so it is a non-negative i64.
        let key = i64::try_from(j).unwrap_or(i64::MAX);
        match &self.spec.kind {
            // A `GeneratedTable` never holds `Rope`, which `new` refuses.
            TableKind::Range { .. } | TableKind::Rope { .. } => 0,
            TableKind::Requant {
                b,
                r_by_class,
                lo,
                hi,
            } => {
                let class = key >> b;
                let k = (key & ((1 << b) - 1)) - (1 << (b - 1));
                requant(k, shift_of(r_by_class, class), *lo, *hi)
            }
            TableKind::Pow2 { .. } => 1 << key,
            TableKind::Bitwise8 { op } => bitwise(*op, key),
            TableKind::AesSbox => i64::from(gf256::sbox(low_byte(key))),
            TableKind::Gf256Mul { c } => i64::from(gf256::mul(*c, low_byte(key))),
            TableKind::Exp { .. } | TableKind::Exp2Frac { .. } => usize::try_from(j)
                .ok()
                .and_then(|j| self.core.get(j))
                .copied()
                .unwrap_or(0),
            TableKind::Silu { b, r_by_class, .. } => {
                let k = (key & ((1 << b) - 1)) - (1 << (b - 1));
                let class = usize::try_from(key >> b).unwrap_or(usize::MAX);
                self.silu_core(*b, r_by_class, class, k)
            }
        }
    }

    /// The core value of `Silu`'s class `class` at the key `k`: its `h = k >> r_c`, in the class's
    /// run of the core, which holds every `h` of every class in order.
    fn silu_core(&self, b: u8, r_by_class: &[u8], class: usize, k: i64) -> i64 {
        let start: u64 = r_by_class
            .iter()
            .take(class)
            .map(|&r| 1u64 << (b - r))
            .sum();
        let r = r_by_class.get(class).copied().unwrap_or(0);
        // `h + 2^{b−1−r}` is `h`'s position among the class's `2^{b−r}` values.
        let offset = (k >> r) + (1i64 << (b - 1 - r));
        u64::try_from(offset)
            .ok()
            .and_then(|o| usize::try_from(start + o).ok())
            .and_then(|i| self.core.get(i))
            .copied()
            .unwrap_or(0)
    }
}

impl Table for GeneratedTable {
    fn spec(&self) -> &TableSpec {
        &self.spec
    }

    fn row_count(&self) -> u64 {
        self.rows
    }

    fn row(&self, index: u64) -> Option<(Goldilocks, Goldilocks)> {
        if index >> self.spec.log2_rows != 0 {
            return None;
        }
        let j = if index < self.rows { index } else { 0 };
        Some((
            Goldilocks::from_u64_reduced(j),
            Goldilocks::from_i64(self.value(j)),
        ))
    }

    fn index(&self, key: i64, class: u8) -> Option<u64> {
        let (index, inside) = match &self.spec.kind {
            TableKind::Requant { b, r_by_class, .. } | TableKind::Silu { b, r_by_class, .. } => {
                let half = 1i64 << (b - 1);
                let classes = i64::try_from(r_by_class.len()).unwrap_or(0);
                let c = i64::from(class);
                let inside = (-half <= key) & (key < half) & (c < classes);
                // Computed before the check, so a key far outside must not overflow.
                (key.wrapping_add(half).wrapping_add(c << b), inside)
            }
            _ => {
                let inside =
                    (0 <= key) & (key < i64::try_from(self.rows).unwrap_or(0)) & (class == 0);
                (key, inside)
            }
        };
        u64::try_from(index).ok().filter(|_| inside)
    }

    fn lookup(&self, key: i64, class: u8) -> Option<i64> {
        let row = self.index(key, class)?;
        Some(match &self.spec.kind {
            TableKind::Range { .. } | TableKind::Rope { .. } => 0,
            TableKind::Requant {
                r_by_class, lo, hi, ..
            } => requant(key, shift_of(r_by_class, i64::from(class)), *lo, *hi),
            TableKind::Pow2 { .. } => 1 << key,
            TableKind::Bitwise8 { op } => bitwise(*op, key),
            TableKind::AesSbox => i64::from(gf256::sbox(low_byte(key))),
            TableKind::Gf256Mul { c } => i64::from(gf256::mul(*c, low_byte(key))),
            TableKind::Exp { .. } | TableKind::Exp2Frac { .. } | TableKind::Silu { .. } => {
                self.value(row)
            }
        })
    }
}

/// `T` for `kind`, after checking its parameters against RFC-0008 'Table specifications'.
fn row_count(kind: &TableKind) -> Result<u64, TableError> {
    let refuse = |kind, rule| Err(TableError::Parameter { kind, rule });
    match kind {
        TableKind::Range { bits } => {
            if !(1..=32).contains(bits) {
                return refuse("Range", "1 ≤ bits ≤ 32");
            }
            Ok(1 << bits)
        }
        TableKind::Requant {
            b,
            r_by_class,
            lo,
            hi,
        } => {
            let classes = r_by_class.len();
            if !(1..=4).contains(&classes) {
                return refuse("Requant", "one to four classes");
            }
            let classes = u64::try_from(classes).unwrap_or(u64::MAX);
            if *b < 1 || u32::from(*b) + u32::from(ceil_log2(classes)) > 32 {
                return refuse("Requant", "b ≥ 1 and b + ⌈log2 n⌉ ≤ 32 for n classes");
            }
            if r_by_class.iter().any(|r| !(1..=63).contains(r)) {
                return refuse("Requant", "1 ≤ r_c ≤ 63 for every class");
            }
            if lo > hi {
                return refuse("Requant", "lo ≤ hi");
            }
            Ok(classes << b)
        }
        TableKind::Pow2 { max_exp } => {
            if !(1..=63).contains(max_exp) {
                return refuse("Pow2", "1 ≤ max_exp ≤ 63");
            }
            Ok(u64::from(*max_exp))
        }
        TableKind::Bitwise8 { .. } => Ok(1 << 16),
        TableKind::AesSbox | TableKind::Gf256Mul { .. } => Ok(1 << 8),
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        } => {
            if !(1..=32).contains(key_bits) {
                return refuse("Exp", "1 ≤ key_bits ≤ 32");
            }
            if *sigma > 32 {
                return refuse("Exp", "σ ≤ 32");
            }
            if scale.num() < 1 {
                return refuse("Exp", "a positive scale");
            }
            Ok(1 << key_bits)
        }
        TableKind::Exp2Frac {
            frac_bits,
            out_bits,
        } => {
            if !(1..=16).contains(frac_bits) {
                return refuse("Exp2Frac", "1 ≤ frac_bits ≤ 16");
            }
            if !(1..=56).contains(out_bits) {
                return refuse("Exp2Frac", "1 ≤ out_bits ≤ 56");
            }
            Ok(1 << frac_bits)
        }
        TableKind::Rope { .. } => refuse("Rope", "built as a RopeTable: it is no lookup table"),
        TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        } => {
            let classes = r_by_class.len();
            if !(1..=4).contains(&classes) {
                return refuse("Silu", "one to four classes");
            }
            let classes = u64::try_from(classes).unwrap_or(u64::MAX);
            if *b < 1 || u32::from(*b) + u32::from(ceil_log2(classes)) > 32 {
                return refuse("Silu", "b ≥ 1 and b + ⌈log2 n⌉ ≤ 32 for n classes");
            }
            if r_by_class.iter().any(|r| r >= b) {
                return refuse("Silu", "r_c < b for every class");
            }
            let values: u64 = r_by_class.iter().map(|&r| 1u64 << (b - r)).sum();
            if values > MAX_CORE {
                return refuse("Silu", "at most 2^24 values of h over the classes");
            }
            if !(-64..=0).contains(in_exp) {
                return refuse("Silu", "−64 ≤ in_exp ≤ 0");
            }
            if !(1..=126).contains(&(63 - i32::from(*in_exp) + i32::from(*out_exp))) {
                return refuse("Silu", "1 ≤ 63 − in_exp + out_exp ≤ 126");
            }
            Ok(classes << b)
        }
    }
}

/// The dense core of the fixed-point kinds (RFC-0008 'Fixed-point generators'); empty for the
/// others.
fn generate_core(kind: &TableKind, rows: u64) -> Result<Vec<i64>, TableError> {
    match kind {
        TableKind::Exp { sigma, scale, .. } => {
            // `scale.num() ≥ 1`, checked with the row count.
            let (num, den) = (scale.num().unsigned_abs(), scale.den());
            let tail = zero_tail(rows, *sigma, num, den);
            if tail > MAX_CORE {
                return Err(TableError::Parameter {
                    kind: "Exp",
                    rule: "a zero tail from key 2^24 or earlier",
                });
            }
            Ok((0..tail).map(|k| exp_value(k, *sigma, num, den)).collect())
        }
        TableKind::Exp2Frac {
            frac_bits,
            out_bits,
        } => Ok((0..rows)
            .map(|f| {
                let x = (u128::from(f) * LN2) >> frac_bits;
                // `≤ 2^63 >> (62 − out_bits) = 2^{out_bits + 1} ≤ 2^57`.
                i64::try_from(exp_pos(x) >> (62 - out_bits)).unwrap_or(i64::MAX)
            })
            .collect()),
        TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        } => Ok(r_by_class
            .iter()
            .flat_map(|&r| {
                let half = 1i64 << (b - 1 - r);
                (-half..half).map(move |h| silu_value(h, *in_exp, *out_exp))
            })
            .collect()),
        _ => Ok(Vec::new()),
    }
}

/// `clamp(round(silu(h · 2^{in_exp}) / 2^{out_exp}), −128, 127)` in the fixed point of RFC-0008:
/// the logistic `σ(x)` in Q63 from `E = e^{−|x|}` in Q64, times `h`, rounded half up by a shift
/// of `s = 63 − in_exp + out_exp`. `|h| < 2^31`, so `|x| · 2^64 < 2^95` and `|h| · σ < 2^94`.
fn silu_value(h: i64, in_exp: i8, out_exp: i8) -> i64 {
    let magnitude = u128::from(h.unsigned_abs());
    // `64 + in_exp ∈ [0, 64]`, checked with the row count.
    let (y, n) = exp_neg(magnitude << (64 + i32::from(in_exp)));
    let e = if n < 128 { y >> n } else { 0 };
    let one = 1u128 << 64;
    let sigma = if h >= 0 {
        (1u128 << 127) / (one + e)
    } else {
        (e << 63) / (one + e)
    };
    let v = i128::from(h) * i128::try_from(sigma).unwrap_or(i128::MAX);
    let s = 63 - i32::from(in_exp) + i32::from(out_exp);
    let rounded = (v + (1i128 << (s - 1))) >> s;
    // Clamped to INT8, so the narrowing cannot fail.
    i64::try_from(rounded.clamp(-128, 127)).unwrap_or(0)
}

/// `k · scale` in Q64, or `None` once its integer part reaches `2^20`, where `e^{−k·scale}` is
/// far below every output unit.
fn exp_argument(k: u64, num: u64, den: u64) -> Option<u128> {
    let product = u128::from(k) * u128::from(num);
    let den = u128::from(den);
    let whole = product / den;
    if whole >= 1 << 20 {
        return None;
    }
    // The remainder is below `den < 2^64`, so shifting it by 64 stays below `2^128`.
    let fraction = ((product - whole * den) << 64) / den;
    Some((whole << 64) + fraction)
}

/// Whether key `k` lies in `Exp`'s zero tail: `k · scale ≥ 2^20`, or `e^{−k·scale}` reduced by
/// at least `σ + 2` halvings, so that `2^σ · e^{−k·scale} < 1/4` and every later key is 0 too.
fn in_zero_tail(k: u64, sigma: u8, num: u64, den: u64) -> bool {
    exp_argument(k, num, den).is_none_or(|x| x / LN2 >= u128::from(sigma) + 2)
}

/// The first key of `Exp`'s zero tail, or `rows` when it has none. Keys only grow `k · scale`,
/// so the tail is a suffix and a binary search finds where it starts.
fn zero_tail(rows: u64, sigma: u8, num: u64, den: u64) -> u64 {
    let (mut lo, mut hi) = (0, rows);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if in_zero_tail(mid, sigma, num, den) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// `round(2^σ · e^{−k·scale})`, rounding half up, outside the zero tail.
fn exp_value(k: u64, sigma: u8, num: u64, den: u64) -> i64 {
    let Some(x) = exp_argument(k, num, den) else {
        return 0;
    };
    let (y, n) = exp_neg(x);
    if n >= u128::from(sigma) + 2 {
        return 0;
    }
    // `y · 2^σ ≤ 2^96` and `2^{63+n} ≤ 2^96`: no overflow, and the result is at most `2^32`.
    let rounded = ((y << sigma) + (1u128 << (63 + n))) >> (64 + n);
    i64::try_from(rounded).unwrap_or(i64::MAX)
}

/// `⌈log2 n⌉` for `1 ≤ n ≤ 2^32`.
fn ceil_log2(n: u64) -> u8 {
    // At most 64, so the narrowing cannot fail.
    u8::try_from(u64::BITS - (n - 1).leading_zeros()).unwrap_or(u8::MAX)
}

/// `rq(k, r, lo, hi) = clamp(⌊(k + 2^{r−1}) / 2^r⌋, lo, hi)` of RFC-0005 'Integer semantics', for
/// `|k| < 2^31` and `1 ≤ r ≤ 63`: the sum stays below `2^62 + 2^31`, and an arithmetic shift is
/// floor division. The clamp selects instead of branching.
fn requant(k: i64, r: u8, lo: i32, hi: i32) -> i64 {
    let shifted = (k + (1i64 << (r - 1))) >> r;
    let (lo, hi) = (i64::from(lo), i64::from(hi));
    let floored = select_unpredictable(shifted < lo, lo, shifted);
    select_unpredictable(floored > hi, hi, floored)
}

/// The shift of `class`, read without an index that depends on it: every entry is visited and
/// the matching one kept. Callers have checked the class; one past the end would yield 1.
fn shift_of(r_by_class: &[u8], class: i64) -> u8 {
    r_by_class
        .iter()
        .zip(0i64..)
        .fold(1, |kept, (&r, c)| select_unpredictable(c == class, r, kept))
}

/// `a op b` for the key `a + 2^8 · b`.
fn bitwise(op: BitOp, key: i64) -> i64 {
    let (a, b) = (key & 0xff, (key >> 8) & 0xff);
    match op {
        BitOp::Xor => a ^ b,
        BitOp::And => a & b,
    }
}

/// The key's low byte, for the byte tables, whose keys have been checked to be bytes.
fn low_byte(key: i64) -> u8 {
    key.to_le_bytes()[0]
}
