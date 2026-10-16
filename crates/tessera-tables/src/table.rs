// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tables generated from their kind: the rows the LogUp prover commits to, and the witness-side
//! lookup of a key (RFC-0008 'Interfaces').

use core::hint::select_unpredictable;

use tessera_codec::Encode;
use tessera_field::{Goldilocks, PrimeField};
use tessera_hash::{Digest, Domain, Hasher};

use crate::gf256;
use crate::{BitOp, TableError, TableId, TableKind, TableSpec};

/// A lookup table of RFC-0008: `T` rows of `(key, value)`, padded to `2^log2_rows` with copies
/// of row 0.
///
/// [`index`](Table::index) and [`lookup`](Table::lookup) take a witness's key, which is secret
/// (05 §3). They compute the row and the value arithmetically, with no memory access that
/// depends on the key, and the class's shift is selected without indexing by it (INV-ARCH-2).
/// Whether the key is in the table is the one outcome that depends on it, and an honest
/// witness's keys always are (INV-DM-2).
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

/// A table of an arithmetic kind: `Range`, `Requant`, `Pow2`, `Bitwise8`, `AesSbox` or
/// `Gf256Mul`. Rows are computed on demand, so building one costs nothing but its digest, which
/// enumerates the value column of the kinds that have one (at most `2^16` rows).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArithmeticTable {
    spec: TableSpec,
    rows: u64,
}

impl ArithmeticTable {
    /// The table of `kind` with index `id`: its parameters checked against RFC-0008 'Table
    /// specifications', its `log2_rows` and digest computed.
    ///
    /// # Errors
    ///
    /// [`TableError::Parameter`] for parameters the kind does not allow.
    pub fn new(id: TableId, kind: TableKind) -> Result<Self, TableError> {
        let rows = row_count(&kind)?;
        let log2_rows = ceil_log2(rows);
        let mut table = Self {
            spec: TableSpec {
                id,
                kind,
                log2_rows,
                digest: [0; 32],
            },
            rows,
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
    /// and digests'). The arithmetic kinds have no piece parameters. The core values are the
    /// value column of the kinds the verifier enumerates, and nothing for `Range` and `Requant`,
    /// whose values are closed forms of their parameters.
    fn digest(&self) -> Digest {
        let mut hasher = Hasher::new(Domain::Table);
        self.spec.id.encode_to(&mut hasher);
        self.spec.kind.encode_to(&mut hasher);
        self.spec.log2_rows.encode_to(&mut hasher);
        // TCE sequences: a u64 length, then the elements (RFC-0003).
        0u64.encode_to(&mut hasher);
        match self.spec.kind {
            TableKind::Range { .. } | TableKind::Requant { .. } => 0u64.encode_to(&mut hasher),
            TableKind::Pow2 { .. }
            | TableKind::Bitwise8 { .. }
            | TableKind::AesSbox
            | TableKind::Gf256Mul { .. } => {
                self.rows.encode_to(&mut hasher);
                for j in 0..self.rows {
                    Goldilocks::from_i64(self.value(j)).encode_to(&mut hasher);
                }
            }
        }
        hasher.finalize()
    }

    /// The value of row `j < T`, the row whose key is `j`.
    fn value(&self, j: u64) -> i64 {
        // `j < T ≤ 2^32`, so it is a non-negative i64.
        let key = i64::try_from(j).unwrap_or(i64::MAX);
        match &self.spec.kind {
            TableKind::Range { .. } => 0,
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
        }
    }
}

impl Table for ArithmeticTable {
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
            TableKind::Requant { b, r_by_class, .. } => {
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
        self.index(key, class)?;
        Some(match &self.spec.kind {
            TableKind::Range { .. } => 0,
            TableKind::Requant {
                r_by_class, lo, hi, ..
            } => requant(key, shift_of(r_by_class, i64::from(class)), *lo, *hi),
            TableKind::Pow2 { .. } => 1 << key,
            TableKind::Bitwise8 { op } => bitwise(*op, key),
            TableKind::AesSbox => i64::from(gf256::sbox(low_byte(key))),
            TableKind::Gf256Mul { c } => i64::from(gf256::mul(*c, low_byte(key))),
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
    }
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
