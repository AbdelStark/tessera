// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The requantization gadgets `Requant1` and `RequantSplit` (RFC-0009, RFC-0028) and
//! `RequantWide` (RFC-0024), from the compiler's selection and the tables it names.
//!
//! Each proves `y = clamp(⌊(z + 2^{r_c−1}) / 2^{r_c}⌋, lo, hi)` for a key `z` of `b` signed bits
//! and the shift `r_c` of its hidden class `c`. With `z_off = z + 2^{b−1}` and the reduced
//! tables' class packing:
//!
//! - **`Requant1`** looks up `(z_off + 2^b·c, y)` in `Requant { b, r_by_class, lo, hi }`.
//! - **`RequantSplit`** commits `z_lo` and `y`, and looks up `z_lo` in `Range { s }` and
//!   `(z_hi + 2^{b−s}·c, y)` in `Requant { b − s, r_by_class − s, lo, hi }`, where
//!   `z_hi = (z_off − z_lo)·2^{−s}`. `z_hi = ⌊z_off / 2^s⌋` is already the offset encoding of
//!   `⌊z / 2^s⌋` in `b − s` bits, so the reduced table's packing adds only the class. Its witness
//!   is unique only when `n·2^b < P` for its `n` classes (RFC-0009 Lemma 2).
//! - **`RequantWide`** commits `z_lo` (when `s ≥ 1`), `h0`, `g` and `y`, with
//!   `w_off = (z_off − z_lo)·2^{−s}`, `h1_off = (w_off − h0)·2^{−t}` and `w' = h0 + 2^t·g`. It
//!   looks up `z_lo` in `Range { s }`, `h0` in `Range { t }`, `h1_off` in `Range { k }`,
//!   `(2·h1_off, g)` in `Requant { k + 1, [1], −2, 1 }` and `(w' + 2^{t+1} + 2^{t+2}·c, y)` in
//!   `Requant { t + 2, r_by_class − s, lo, hi }`.
//!
//! A gadget's class term is present only when its output table has several classes: a site of
//! one class has no class column (RFC-0006 'Llama lowering').

use tessera_compiler::Gadget;
use tessera_field::{Goldilocks, PrimeField};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

use crate::constraints::{affine, inv_pow2, pow2, tensor};
use crate::{Committed, Constraints, GadgetError, Lookup, Term, Virtual};

/// The bound of RFC-0009 Lemma 2 and RFC-0024 Lemma 11 on a key: its parts stay true integers.
const MAX_KEY_BITS: u32 = 62;

/// The protocol's prime, Goldilocks' `P = 2^64 − 2^32 + 1`, which bounds a split key of `n`
/// classes: `n·2^b < P` (RFC-0009 Lemma 2).
const PRIME: u128 = Goldilocks::MODULUS as u128;

/// A requantization gadget, `Requant1`, `RequantSplit` or `RequantWide`, with the tables its
/// lookups read, regenerated from their specifications (RFC-0009, RFC-0024).
///
/// [`new`](Self::new) builds it from the compiler's [`Gadget`] and checks that the tables are
/// the ones the gadget reads; [`constraints`](Self::constraints) describes its lookups, and
/// [`witness`](Self::witness) computes its committed tensors from the keys and classes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requant {
    pub(crate) form: Form,
}

/// A requantization gadget's parameters and tables. Widths are in bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    /// `Requant1`.
    One {
        /// `b`, the key's signed width.
        b: u32,
        /// `Requant { b, r_by_class, lo, hi }`.
        table: GeneratedTable,
    },
    /// `RequantSplit`.
    Split {
        /// `b`.
        b: u32,
        /// `s`, the low part.
        split: u32,
        /// `Range { s }`.
        low: GeneratedTable,
        /// `Requant { b − s, r_by_class − s, lo, hi }`.
        high: GeneratedTable,
    },
    /// `RequantWide`, boxed: it holds five tables.
    Wide(Box<Wide>),
}

/// `RequantWide`'s parameters and tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Wide {
    /// `b = s + t + k`.
    pub(crate) b: u32,
    /// `s`, the low part, below every class's rounding bit.
    pub(crate) split: u32,
    /// `t`, the middle part.
    pub(crate) middle: u32,
    /// `k`, the top part.
    pub(crate) top_bits: u32,
    /// `Range { s }`, or `None` when `s = 0`.
    pub(crate) low: Option<GeneratedTable>,
    /// `Range { t }`.
    pub(crate) mid: GeneratedTable,
    /// `Range { k }`.
    pub(crate) top: GeneratedTable,
    /// `Requant { k + 1, [1], −2, 1 }`.
    pub(crate) sign: GeneratedTable,
    /// `Requant { t + 2, r_by_class − s, lo, hi }`.
    pub(crate) high: GeneratedTable,
}

impl Requant {
    /// Requested owned heap of this public template, including RequantWide's
    /// boxed descriptor and every owned table payload, even with repeated ids.
    /// Excludes inline `Self`, constructor scratch and secret witnesses (RFC-0018).
    /// Returns `None` on size overflow and allocates nothing.
    pub fn heap_bytes(&self) -> Option<usize> {
        match &self.form {
            Form::One { table, .. } => crate::storage::table(table),
            Form::Split { low, high, .. } => {
                crate::storage::table(low)?.checked_add(crate::storage::table(high)?)
            }
            Form::Wide(wide) => wide
                .low
                .iter()
                .chain([&wide.mid, &wide.top, &wide.sign, &wide.high])
                .try_fold(size_of::<Wide>(), |sum, table| {
                    sum.checked_add(crate::storage::table(table)?)
                }),
        }
    }

    /// The gadget the compiler selected, `gadget`, with its tables among `tables`.
    ///
    /// Each table must be among the specifications and regenerate from its kind, and be the one
    /// the gadget reads there:
    ///
    /// - `Requant1` reads a `Requant` table, of key width `b`.
    /// - `RequantSplit` reads `Range { s }` and a `Requant` table of `b − s` bits and `n` classes,
    ///   with `b ≤ 62` and `n·2^b < P` (Lemma 2). Of keys of at most 62 bits, `n·2^b < P`
    ///   excludes only four classes at 62 bits, which the compiler gives `RequantWide`. Its
    ///   reduced shifts `r_c − s` are at least 1, the table's own rule, which is Lemma 2's
    ///   `s ≤ r_c − 1`.
    /// - `RequantWide` reads `Range { s }` exactly when `s ≥ 1`, then `Range { t }`, `Range { k }`,
    ///   `Requant { k + 1, [1], −2, 1 }` and `Requant { t + 2, r_c − s, lo, hi }`, with
    ///   `b = s + t + k ≤ 62` and `t = (r_max − s) + ⌈log2 M⌉`, `M = max(|lo|, |hi|, 1)`, past
    ///   which every class saturates (Lemma 11).
    ///
    /// # Errors
    ///
    /// In this order: [`GadgetError::Shape`] for a gadget that is no requantization;
    /// [`GadgetError::Table`] for a table that is not among `tables`, then
    /// [`GadgetError::Spec`] for one that does not regenerate, table by table in the gadget's
    /// lookup order, each checked against its role ([`GadgetError::Table`]) as it is read; then
    /// [`GadgetError::Shape`] for a key over 62 bits, a `RequantSplit` with `n·2^b ≥ P` or, for
    /// `RequantWide`, a middle part of another width than RFC-0024's.
    ///
    /// `P` is the protocol's prime, Goldilocks'. Over the test field `F65537`, a gadget this
    /// accepts may break Lemma 2's bound for that field's smaller prime, as the soundness
    /// calibration does on purpose.
    pub fn new(gadget: &Gadget, tables: &[TableSpec]) -> Result<Self, GadgetError> {
        let table = |id: TableId| -> Result<GeneratedTable, GadgetError> {
            let spec = tables
                .iter()
                .find(|spec| spec.id == id)
                .ok_or(GadgetError::Table {
                    table: id,
                    rule: "the specifications hold every table the gadget names",
                })?;
            GeneratedTable::from_spec(spec).map_err(|error| GadgetError::Spec { table: id, error })
        };
        let form = match *gadget {
            Gadget::Requant1 { table: id } => {
                let table = table(id)?;
                let TableKind::Requant { b, .. } = table.spec().kind else {
                    return Err(GadgetError::Table {
                        table: id,
                        rule: "Requant1 reads a Requant table",
                    });
                };
                Form::One {
                    b: u32::from(b),
                    table,
                }
            }
            Gadget::RequantSplit { split, low, high } => {
                let low = range(
                    table(low)?,
                    split,
                    "RequantSplit's low part reads Range { s }",
                )?;
                let high = table(high)?;
                let TableKind::Requant {
                    b: reduced,
                    ref r_by_class,
                    ..
                } = high.spec().kind
                else {
                    return Err(GadgetError::Table {
                        table: high.spec().id,
                        rule: "RequantSplit's high part reads a Requant table",
                    });
                };
                // One to four: the table regenerated.
                let classes = r_by_class.len() as u128;
                let (split, b) = (u32::from(split), u32::from(reduced) + u32::from(split));
                if b > MAX_KEY_BITS {
                    return Err(GadgetError::Shape {
                        rule: "a split key has at most 62 bits (RFC-0009 Lemma 2)",
                    });
                }
                // Lemma 2: the lookups give `z_lo + 2^s·j ≡ z_off + 2^b·c (mod P)` for the high
                // key's row `j`, both sides in `[0, n·2^b)`. They are equal, so that `z_lo` is
                // the true low part, only when they cannot differ by `P`. At `n = 4, b = 62` they
                // can: `z_lo ± 1` wraps the key to the table's other end.
                if classes << b >= PRIME {
                    return Err(GadgetError::Shape {
                        rule: "a split key of n classes has n·2^b < P (RFC-0009 Lemma 2)",
                    });
                }
                Form::Split {
                    b,
                    split,
                    low,
                    high,
                }
            }
            Gadget::RequantWide {
                split,
                middle,
                low,
                mid,
                top,
                sign,
                high,
            } => {
                let low = match (split, low) {
                    (0, None) => None,
                    (1.., Some(id)) => Some(range(
                        table(id)?,
                        split,
                        "RequantWide's low part reads Range { s }",
                    )?),
                    _ => {
                        return Err(GadgetError::Shape {
                            rule: "RequantWide's low part has a Range table exactly when s ≥ 1",
                        });
                    }
                };
                let mid = range(
                    table(mid)?,
                    middle,
                    "RequantWide's middle part reads Range { t }",
                )?;
                let top = table(top)?;
                let TableKind::Range { bits: top_bits } = top.spec().kind else {
                    return Err(GadgetError::Table {
                        table: top.spec().id,
                        rule: "RequantWide's top part reads a Range table",
                    });
                };
                let sign = table(sign)?;
                let saturation = TableKind::Requant {
                    b: top_bits + 1,
                    r_by_class: vec![1],
                    lo: -2,
                    hi: 1,
                };
                if sign.spec().kind != saturation {
                    return Err(GadgetError::Table {
                        table: sign.spec().id,
                        rule: "RequantWide's saturation reads Requant { k + 1, [1], −2, 1 }",
                    });
                }
                let high = table(high)?;
                let TableKind::Requant {
                    b: reduced,
                    ref r_by_class,
                    lo,
                    hi,
                } = high.spec().kind
                else {
                    return Err(GadgetError::Table {
                        table: high.spec().id,
                        rule: "RequantWide's output reads a Requant table",
                    });
                };
                if u32::from(reduced) != u32::from(middle) + 2 {
                    return Err(GadgetError::Table {
                        table: high.spec().id,
                        rule: "RequantWide's output reads Requant { t + 2, r_c − s, lo, hi }",
                    });
                }
                let (split, middle, top_bits) =
                    (u32::from(split), u32::from(middle), u32::from(top_bits));
                let b = split + middle + top_bits;
                if b > MAX_KEY_BITS {
                    return Err(GadgetError::Shape {
                        rule: "a wide key has at most 62 bits (RFC-0024 Lemma 11)",
                    });
                }
                // Past |w| ≥ 2^t every class saturates: 2^{t − (r_c − s)} ≥ M for the largest
                // reduced shift, the bound Lemma 11's value proof reads.
                let bound = u64::from(hi.unsigned_abs().max(lo.unsigned_abs()).max(1));
                let reduced_max = r_by_class.iter().copied().max().map_or(0, u32::from);
                if middle != reduced_max + bound.next_power_of_two().trailing_zeros() {
                    return Err(GadgetError::Shape {
                        rule: "RequantWide's middle part has t = (r_max − s) + ⌈log2 M⌉ bits (RFC-0024)",
                    });
                }
                Form::Wide(Box::new(Wide {
                    b,
                    split,
                    middle,
                    top_bits,
                    low,
                    mid,
                    top,
                    sign,
                    high,
                }))
            }
            _ => {
                return Err(GadgetError::Shape {
                    rule: "a requantization gadget: Requant1, RequantSplit or RequantWide",
                });
            }
        };
        Ok(Self { form })
    }

    /// `b`, the signed width of the keys the gadget takes: `[−2^{b−1}, 2^{b−1})`.
    pub fn key_bits(&self) -> u32 {
        match &self.form {
            Form::One { b, .. } | Form::Split { b, .. } => *b,
            Form::Wide(wide) => wide.b,
        }
    }

    /// `n`, the number of shift classes, from one to four: a class is an index below it.
    pub fn classes(&self) -> usize {
        match &self.output().spec().kind {
            TableKind::Requant { r_by_class, .. } => r_by_class.len(),
            // `new` checked that the output table is a `Requant` table.
            _ => 1,
        }
    }

    /// The tables the gadget's lookups read, each once, in the order of its lookups.
    pub fn tables(&self) -> Vec<&GeneratedTable> {
        let all: Vec<&GeneratedTable> = match &self.form {
            Form::One { table, .. } => vec![table],
            Form::Split { low, high, .. } => vec![low, high],
            Form::Wide(wide) => wide
                .low
                .iter()
                .chain([&wide.mid, &wide.top, &wide.sign, &wide.high])
                .collect(),
        };
        let mut distinct: Vec<&GeneratedTable> = Vec::with_capacity(all.len());
        for table in all {
            if distinct.iter().all(|t| t.spec().id != table.spec().id) {
                distinct.push(table);
            }
        }
        distinct
    }

    /// The table that returns `y`.
    pub(crate) fn output(&self) -> &GeneratedTable {
        match &self.form {
            Form::One { table, .. } => table,
            Form::Split { high, .. } => high,
            Form::Wide(wide) => &wide.high,
        }
    }

    /// The gadget's constraints over `F` (RFC-0009, RFC-0024), in the order of its section:
    ///
    /// - **`Requant1`** commits `y`; its one lookup is `(z + 2^{b−1} + 2^b·c, y)`.
    /// - **`RequantSplit`** commits `z_lo` and `y`. Its virtual values are `z_off = z + 2^{b−1}`
    ///   and `z_hi = 2^{−s}·z − 2^{−s}·z_lo + 2^{b−1−s}`. Its lookups are `z_lo`, then
    ///   `(z_hi + 2^{b−s}·c, y)`.
    /// - **`RequantWide`** commits `z_lo` when `s ≥ 1`, `h0`, `g` and `y`. Its virtual values are
    ///   `z_off`, `w_off = 2^{−s}·(z − z_lo) + 2^{b−1−s}`,
    ///   `h1_off = 2^{−s−t}·(z − z_lo) − 2^{−t}·h0 + 2^{k−1}`, `h1 = h1_off − 2^{k−1}` and
    ///   `w' = h0 + 2^t·g`. Its lookups are `z_lo` when `s ≥ 1`, `h0`, `h1_off`,
    ///   `(2·h1_off, g)` and `(w' + 2^{t+1} + 2^{t+2}·c, y)`.
    ///
    /// The class term `2^{b'}·c` of a reduced table of `b'` key bits is present when that table
    /// has several classes. Each gadget's lemma takes `c` to be an integer in `[0, n)`, the
    /// classes' domain in RFC-0006 'Range analysis', and the lookups alone do not establish it
    /// (RFC-0009 'Requant1').
    pub fn constraints<F: PrimeField>(&self) -> Constraints<F> {
        let classed = self.classes() > 1;
        // The class packing of a table of `bits` key bits, when it has several classes.
        let class = |bits: u32| -> Vec<(F, Term)> {
            match classed {
                true => vec![(pow2(bits), Term::Class)],
                false => Vec::new(),
            }
        };
        let id = |table: &GeneratedTable| table.spec().id;
        let input = Term::Input;
        let z_lo = Term::Committed(Committed::ZLo);
        match &self.form {
            Form::One { b, table } => {
                let mut terms = vec![(F::ONE, input)];
                terms.extend(class(*b));
                Constraints {
                    committed: vec![Committed::Y],
                    virtuals: Vec::new(),
                    lookups: vec![Lookup {
                        table: id(table),
                        key: affine(pow2(b - 1), &terms),
                        value: Some(tensor(Committed::Y)),
                    }],
                }
            }
            Form::Split {
                b,
                split,
                low,
                high,
            } => {
                let inverse = inv_pow2::<F>(*split);
                let z_hi = [(inverse, input), (-inverse, z_lo)];
                let mut high_key = z_hi.to_vec();
                high_key.extend(class(b - split));
                Constraints {
                    committed: vec![Committed::ZLo, Committed::Y],
                    virtuals: vec![
                        (Virtual::ZOff, affine(pow2(b - 1), &[(F::ONE, input)])),
                        (Virtual::ZHi, affine(pow2(b - 1 - split), &z_hi)),
                    ],
                    lookups: vec![
                        Lookup {
                            table: id(low),
                            key: tensor(Committed::ZLo),
                            value: None,
                        },
                        Lookup {
                            table: id(high),
                            key: affine(pow2(b - 1 - split), &high_key),
                            value: Some(tensor(Committed::Y)),
                        },
                    ],
                }
            }
            Form::Wide(wide) => {
                let Wide {
                    b,
                    split: s,
                    middle: t,
                    top_bits: k,
                    low,
                    mid,
                    top,
                    sign,
                    high,
                } = &**wide;
                let (b, s, t, k) = (*b, *s, *t, *k);
                let h0 = Term::Committed(Committed::H0);
                let g = Term::Committed(Committed::G);
                // `scale·(z − z_lo)`, without `z_lo` when `s = 0`.
                let reduced = |scale: F| -> Vec<(F, Term)> {
                    match low {
                        Some(_) => vec![(scale, input), (-scale, z_lo)],
                        None => vec![(scale, input)],
                    }
                };
                let w_off = reduced(inv_pow2(s));
                let mut h1 = reduced(inv_pow2(s + t));
                h1.push((-inv_pow2::<F>(t), h0));
                let two = F::ONE + F::ONE;
                let doubled: Vec<(F, Term)> = h1.iter().map(|&(a, term)| (two * a, term)).collect();
                let w_prime = [(F::ONE, h0), (pow2(t), g)];
                let mut high_key = w_prime.to_vec();
                high_key.extend(class(t + 2));
                let mut committed = Vec::with_capacity(4);
                let mut lookups = Vec::with_capacity(5);
                if let Some(low) = low {
                    committed.push(Committed::ZLo);
                    lookups.push(Lookup {
                        table: id(low),
                        key: tensor(Committed::ZLo),
                        value: None,
                    });
                }
                committed.extend([Committed::H0, Committed::G, Committed::Y]);
                lookups.extend([
                    Lookup {
                        table: id(mid),
                        key: tensor(Committed::H0),
                        value: None,
                    },
                    Lookup {
                        table: id(top),
                        key: affine(pow2(k - 1), &h1),
                        value: None,
                    },
                    // `pack(0, 2·h1) = 2·h1 + 2^k = 2·h1_off`.
                    Lookup {
                        table: id(sign),
                        key: affine(pow2(k), &doubled),
                        value: Some(tensor(Committed::G)),
                    },
                    Lookup {
                        table: id(high),
                        key: affine(pow2(t + 1), &high_key),
                        value: Some(tensor(Committed::Y)),
                    },
                ]);
                Constraints {
                    committed,
                    virtuals: vec![
                        (Virtual::ZOff, affine(pow2(b - 1), &[(F::ONE, input)])),
                        (Virtual::WOff, affine(pow2(b - 1 - s), &w_off)),
                        (Virtual::H1Off, affine(pow2(k - 1), &h1)),
                        (Virtual::H1, affine(F::ZERO, &h1)),
                        (Virtual::WPrime, affine(F::ZERO, &w_prime)),
                    ],
                    lookups,
                }
            }
        }
    }
}

/// `table`, after checking that it is `Range { bits }`.
fn range(
    table: GeneratedTable,
    bits: u8,
    rule: &'static str,
) -> Result<GeneratedTable, GadgetError> {
    match table.spec().kind == (TableKind::Range { bits }) {
        true => Ok(table),
        false => Err(GadgetError::Table {
            table: table.spec().id,
            rule,
        }),
    }
}
