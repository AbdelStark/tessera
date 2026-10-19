// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gadget selection and table sizing (RFC-0006 'Gadget selection', RFC-0009, RFC-0024).
//!
//! Every gadget-backed node gets its gadget, numbered in node order. A `Requant` takes one of
//! three by its key's signed width `b`, from range analysis, and its `n` shift classes, whose
//! index takes `⌈log2 n⌉` more key bits in a table. The first that applies is cheapest:
//!
//! - **`Requant1`** reads `Requant { b, r_by_class, lo, hi }` once, when `b + ⌈log2 n⌉ ≤ B_max`.
//! - **`RequantSplit`** sends the low `s` bits of the offset key to `Range { s }` and `⌊z / 2^s⌋`
//!   to `Requant { b − s, r_by_class − s, lo, hi }`, from `s = b + ⌈log2 n⌉ − B_max` at least, at
//!   the point that balances the two tables (RFC-0028). The nested floor is the requantization
//!   only when `s ≤ r_c − 1` for every class, the halves are true integers only when `b ≤ 62`,
//!   the low part's table must fit too, `s ≤ B_max`, and the low part is unique only when
//!   `n·2^b < P` (RFC-0009 Lemma 2).
//! - **`RequantWide`** takes a key the split cannot (RFC-0024): one too wide for its shift, or of
//!   four classes at 62 bits, where `n·2^b ≥ P`. Below every class's rounding bit,
//!   `s = min(r_min − 1, B_max)` low bits go to `Range { s }`. The rest, `w = ⌊z / 2^s⌋`, splits
//!   into a middle `h0` of `t` bits and a top `h1` of `k = b − s − t` bits, each checked by a
//!   `Range` table. `t` is the least width past which every class saturates. The top part's only
//!   effect is its saturation `g = clamp(h1, −2, 1)`, read from `Requant { k + 1, [1], −2, 1 }` at
//!   the key `2·h1`, so `Requant { t + 2, r_by_class − s, lo, hi }` of `h0 + 2^t · g` is the
//!   output. Its keys are true integers when `b ≤ 62`, and its tables must fit `B_max`.
//!
//! Any other key is `ERR-COMPILE-RANGE`. A `TableMap`, `Bitwise` or `SoftmaxRow` reads a table of
//! its own, which must hold at most `2^{B_max}` rows, since those lookups have no split.
//!
//! Tables are shared by definition: a requantization table is its kind with `(b, r_by_class, lo,
//! hi)`, and two nodes asking for the same one name one table. New tables follow the lowering's,
//! numbered in order of first use, so the selection is a function of the graph (INV-IR-4).

use tessera_ir::{GadgetId, GadgetPolicy, Graph, Node, NodeId, Op, TableId};
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

use crate::{CompileError, HALF_FIELD, Range, Ranges};

/// The bound of RFC-0009 Lemma 2 on a split key: both halves stay true integers.
const SPLIT_KEY_BITS: u32 = 62;

/// The Goldilocks prime `P = 2^64 − 2^32 + 1`, which bounds a split key of `n` classes:
/// `n·2^b < P` (RFC-0009 Lemma 2).
const PRIME: u128 = (2 * HALF_FIELD + 1) as u128;

/// The gadget of a node (RFC-0009).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gadget {
    /// `Requant1`: one lookup of the key and its class.
    Requant1 {
        /// The `Requant` table.
        table: TableId,
    },
    /// `RequantSplit`: the low `split` bits of the offset key, then the rest.
    RequantSplit {
        /// `s`, the low bits.
        split: u8,
        /// `Range { s }`, which checks the low bits.
        low: TableId,
        /// The reduced `Requant` table of `⌊z / 2^s⌋`.
        high: TableId,
    },
    /// `RequantWide`: the low `split` bits, a middle of `middle` bits, and a top part that only
    /// saturates (RFC-0024).
    RequantWide {
        /// `s`, the low bits, below every class's rounding bit.
        split: u8,
        /// `t`, the middle part's bits: the reduced keys that need not saturate.
        middle: u8,
        /// `Range { s }`, which checks the low bits, or `None` when `s = 0`.
        low: Option<TableId>,
        /// `Range { t }`, which checks the middle part.
        mid: TableId,
        /// `Range { k }`, which checks the top part, so that its doubled key is even.
        top: TableId,
        /// `Requant { k + 1, [1], −2, 1 }`, the top part's saturation `clamp(h1, −2, 1)`.
        sign: TableId,
        /// The reduced `Requant` table of the middle part and the saturation.
        high: TableId,
    },
    /// One lookup into the node's table: `TableMap` and `Bitwise`.
    Lookup {
        /// The table.
        table: TableId,
    },
    /// The max certificate and reciprocal bracket, with its `Exp` table.
    SoftmaxRow {
        /// The `Exp` table.
        exp: TableId,
        /// `Range { σ+1 }`, the reciprocal's range.
        reciprocal: TableId,
        /// `Range { L }` for the two slacks' committed limbs, absent for one-limb slacks.
        limb: Option<TableId>,
        /// The virtual top limbs' range; may share the reciprocal or low-limb table.
        top: TableId,
    },
    /// The multi-limb root bracket.
    RsqrtRow,
    /// The one-hot vectors' zero-check and sums.
    OneHot,
    /// The greedy-decoding certificate (RFC-0017).
    Argmax,
    /// The sampling certificate (RFC-0017).
    InverseCdf,
}

/// Every gadget of a graph and every table its lookups read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    /// The gadget-backed nodes in order, each with its gadget: gadget `k` is `GadgetId(k)`.
    pub gadgets: Vec<(NodeId, Gadget)>,
    /// The tables, by id: those the graph names, then gadget auxiliary tables in first-use order.
    pub tables: Vec<TableSpec>,
}

impl Selection {
    /// The gadget of `node`, with its id, or `None` for a node without one.
    pub fn of(&self, node: NodeId) -> Option<(GadgetId, Gadget)> {
        let position = self.gadgets.iter().position(|&(n, _)| n == node)?;
        let id = GadgetId(u32::try_from(position).ok()?);
        Some((id, self.gadgets[position].1))
    }
}

/// The gadget of every gadget-backed node of `graph`, whose ranges are `ranges` and whose lookups
/// name tables among `tables`, under `policy`.
///
/// # Errors
///
/// - [`CompileError::Usage`] for a `B_max` outside `[1, 32]`;
/// - [`CompileError::Range`] for a requantization key no gadget takes, or a table of more than
///   `2^{B_max}` rows;
/// - [`CompileError::Table`] for a table the specifications lack (`ERR-INTERNAL`).
pub fn select(
    graph: &Graph,
    ranges: &Ranges,
    tables: &[TableSpec],
    policy: &GadgetPolicy,
) -> Result<Selection, CompileError> {
    let max = u32::from(policy.max_log2_rows);
    if !(1..=32).contains(&max) {
        return Err(CompileError::Usage {
            rule: "B_max lies in [1, 32], the most a Range table holds",
        });
    }
    let mut selector = Selector {
        graph,
        ranges,
        max,
        limb_bits: policy.limb_bits,
        tables: tables.to_vec(),
    };
    let mut gadgets = Vec::new();
    for node in &graph.nodes {
        if let Some(gadget) = selector.node(node)? {
            gadgets.push((node.id, gadget));
        }
    }
    Ok(Selection {
        gadgets,
        tables: selector.tables,
    })
}

struct Selector<'a> {
    graph: &'a Graph,
    ranges: &'a Ranges,
    max: u32,
    limb_bits: u8,
    tables: Vec<TableSpec>,
}

impl Selector<'_> {
    fn node(&mut self, node: &Node) -> Result<Option<Gadget>, CompileError> {
        Ok(Some(match &node.op {
            Op::Requant { shift, lo, hi } => {
                let shifts = self.graph.shift_classes[shift.0 as usize].clone();
                self.requant(node, shifts, *lo, *hi)?
            }
            Op::TableMap { table } | Op::Bitwise { table } => Gadget::Lookup {
                table: self.sized(node, *table)?,
            },
            Op::SoftmaxRow { exp_table, sigma } => {
                let exp = self.sized(node, *exp_table)?;
                let (reciprocal, limb, top) = self.softmax_ranges(node, *sigma)?;
                Gadget::SoftmaxRow {
                    exp,
                    reciprocal,
                    limb,
                    top,
                }
            }
            Op::RsqrtRow { .. } => Gadget::RsqrtRow,
            Op::OneHot { .. } => Gadget::OneHot,
            Op::Argmax => Gadget::Argmax,
            Op::InverseCdf { .. } => Gadget::InverseCdf,
            _ => return Ok(None),
        }))
    }

    /// RFC-0009 relation 3: R's range, then A and B's shared NonNeg ranges in first-use order.
    fn softmax_ranges(
        &mut self,
        node: &Node,
        sigma: u8,
    ) -> Result<(TableId, Option<TableId>, TableId), CompileError> {
        let l = u32::from(self.limb_bits);
        if l == 0 || l > self.max {
            return Err(CompileError::Range {
                node: node.id,
                input: 0,
                rule: "SoftmaxRow's limb width lies in [1, B_max]",
            });
        }
        let k = u32::from(sigma) + 18;
        let n = k.div_ceil(l);
        let reciprocal = self.range_table(node, u32::from(sigma) + 1)?;
        let limb = if n > 1 {
            Some(self.range_table(node, l)?)
        } else {
            None
        };
        let top = self.range_table(node, k - (n - 1) * l)?;
        Ok((reciprocal, limb, top))
    }

    fn range_table(&mut self, node: &Node, bits: u32) -> Result<TableId, CompileError> {
        if bits > self.max {
            return Err(CompileError::Range {
                node: node.id,
                input: 0,
                rule: "a lookup's table has at most 2^{B_max} rows",
            });
        }
        let table = self.table(node, TableKind::Range { bits: narrow(bits) })?;
        self.sized(node, table)
    }

    /// `Requant1`, `RequantSplit` or `RequantWide` for the key of `node`, by its width, its
    /// shifts and its number of classes.
    fn requant(
        &mut self,
        node: &Node,
        shifts: Vec<u8>,
        lo: i64,
        hi: i64,
    ) -> Result<Gadget, CompileError> {
        let refuse = |rule| CompileError::Range {
            node: node.id,
            input: 0,
            rule,
        };
        let key = match self.ranges.of(node.inputs[0]) {
            Some(Range::Bounded(key)) => key,
            _ => {
                return Err(CompileError::Wrap {
                    node: node.id,
                    input: 0,
                });
            }
        };
        let (lo, hi) = (
            i32::try_from(lo).map_err(|_| refuse("a saturation bound fits 32 bits"))?,
            i32::try_from(hi).map_err(|_| refuse("a saturation bound fits 32 bits"))?,
        );
        let b = signed_width(key.lo, key.hi);
        // The class index's bits: 0, 1, 2 for one, two, three or four classes.
        let class_bits = usize::BITS - (shifts.len().max(1) - 1).leading_zeros();
        if b + class_bits <= self.max {
            let table = self.table(
                node,
                TableKind::Requant {
                    b: narrow(b),
                    r_by_class: shifts,
                    lo,
                    hi,
                },
            )?;
            return Ok(Gadget::Requant1 { table });
        }
        if b > SPLIT_KEY_BITS {
            return Err(refuse("a requantization key has at most 62 bits"));
        }
        // Lemma 2's split points: from the least that fits the high table, `b + ⌈log2 n⌉ − B_max`,
        // to the most that fits the low one and keeps every class's rounding bit,
        // `min(r_min − 1, B_max)`. None: the key takes `RequantWide`.
        let fits = b + class_bits - self.max;
        let least = u32::from(shifts.iter().copied().min().unwrap_or(1));
        let most = (least - 1).min(self.max);
        // Lemma 2 also needs `n·2^b < P`, whatever the split point: `z_lo + 2^s·j` and
        // `z_off + 2^b·c` lie in `[0, n·2^b)`, and must not differ by `P`. Of keys of at most 62
        // bits, it excludes four classes at 62. `RequantWide` takes them: its class term reads
        // only values its lookups pin (RFC-0024 Lemma 11).
        let packed = (shifts.len() as u128) << b;
        if fits > most || packed >= PRIME {
            return self.wide(node, b, class_bits, &shifts, lo, hi);
        }
        // RFC-0028: the balanced point, which minimizes the two tables' rows,
        // `2^s + 2^{b + ⌈log2 n⌉ − s}`. On a tie the low `Range`, which only `s` defines and other
        // sites share, takes the extra bit.
        let split = (b + class_bits).div_ceil(2).clamp(fits, most);
        let low = self.table(
            node,
            TableKind::Range {
                bits: narrow(split),
            },
        )?;
        let high = self.table(
            node,
            TableKind::Requant {
                b: narrow(b - split),
                r_by_class: shifts.iter().map(|&r| r - narrow(split)).collect(),
                lo,
                hi,
            },
        )?;
        Ok(Gadget::RequantSplit {
            split: narrow(split),
            low,
            high,
        })
    }

    /// `RequantWide` for a key of `b ≤ 62` bits that `RequantSplit` cannot take (RFC-0024): its
    /// split points are none, or its `n·2^b` reaches `P`.
    fn wide(
        &mut self,
        node: &Node,
        b: u32,
        class_bits: u32,
        shifts: &[u8],
        lo: i32,
        hi: i32,
    ) -> Result<Gadget, CompileError> {
        let refuse = |rule| CompileError::Range {
            node: node.id,
            input: 0,
            rule,
        };
        let (least, most) = match (shifts.iter().min(), shifts.iter().max()) {
            (Some(&least), Some(&most)) => (u32::from(least), u32::from(most)),
            _ => return Err(refuse("a requantization has a shift class")),
        };
        // The low bits stop below every class's rounding bit, and fit a table.
        let split = (least - 1).min(self.max);
        // Past `|w| ≥ 2^t`, every class saturates: `2^{t − (r_c − s)} ≥ max(hi, −lo)`.
        let bound = u64::from(hi.unsigned_abs().max(lo.unsigned_abs()).max(1));
        let middle = most - split + bound.next_power_of_two().trailing_zeros();
        // `h0 + 2^t · g` lies in `[−2^{t+1}, 2^{t+1})`, with the class index.
        if middle + 2 + class_bits > self.max {
            return Err(refuse(
                "a wide key's reduced table fits 2^{B_max} rows: t + 2 + ⌈log2 n⌉ ≤ B_max",
            ));
        }
        // At least 3 when the split failed `s' ≤ r_min − 1` or `s' ≤ B_max`, where
        // `s' = b + ⌈log2 n⌉ − B_max`, since `t ≤ B_max − 2 − ⌈log2 n⌉` (RFC-0024). At least 2
        // when only `n·2^b < P` failed: then `n = 4`, `b = 62`, `B_max = 32`, `s = 32` and
        // `t ≤ 28`. Saturating, a mistake reaches the generator's refusal of a `Range` of 0 bits.
        let top = b.saturating_sub(split + middle);
        if top + 1 > self.max {
            return Err(refuse(
                "a wide key's top part fits a table of 2^{B_max} rows: k + 1 ≤ B_max",
            ));
        }
        let low = match split {
            0 => None,
            bits => Some(self.table(node, TableKind::Range { bits: narrow(bits) })?),
        };
        let mid = self.table(
            node,
            TableKind::Range {
                bits: narrow(middle),
            },
        )?;
        let top_range = self.table(node, TableKind::Range { bits: narrow(top) })?;
        let sign = self.table(
            node,
            TableKind::Requant {
                b: narrow(top + 1),
                r_by_class: vec![1],
                lo: -2,
                hi: 1,
            },
        )?;
        let high = self.table(
            node,
            TableKind::Requant {
                b: narrow(middle + 2),
                r_by_class: shifts.iter().map(|&r| r - narrow(split)).collect(),
                lo,
                hi,
            },
        )?;
        Ok(Gadget::RequantWide {
            split: narrow(split),
            middle: narrow(middle),
            low,
            mid,
            top: top_range,
            sign,
            high,
        })
    }

    /// `table`, after checking that the graph's specifications hold it with at most `2^{B_max}`
    /// rows.
    fn sized(&self, node: &Node, table: TableId) -> Result<TableId, CompileError> {
        let spec = self
            .tables
            .iter()
            .find(|spec| spec.id == table)
            .ok_or(CompileError::Table {
                node: node.id,
                table,
                rule: "no specification names it",
            })?;
        if u32::from(spec.log2_rows) > self.max {
            return Err(CompileError::Range {
                node: node.id,
                input: 0,
                rule: "a lookup's table has at most 2^{B_max} rows",
            });
        }
        Ok(table)
    }

    /// The table of `kind`: one of the same definition, or a new one numbered after the others.
    fn table(&mut self, node: &Node, kind: TableKind) -> Result<TableId, CompileError> {
        if let Some(spec) = self.tables.iter().find(|spec| spec.kind == kind) {
            return Ok(spec.id);
        }
        let next = self
            .tables
            .iter()
            .map(|spec| spec.id.0 + 1)
            .max()
            .unwrap_or(0);
        let id = TableId(next);
        let table = GeneratedTable::new(id, kind).map_err(|_| CompileError::Table {
            node: node.id,
            table: id,
            rule: "the generator refuses a table selection built",
        })?;
        self.tables.push(table.spec().clone());
        Ok(id)
    }
}

/// The least `b` with `[−2^{b−1}, 2^{b−1})` holding `[lo, hi]`: a `Requant` table's key width.
fn signed_width(lo: i128, hi: i128) -> u32 {
    let width = |x: i128| u128::BITS - u128::try_from(x.max(0)).unwrap_or(0).leading_zeros();
    // `2^{b−1} > hi` and `2^{b−1} ≥ −lo`, that is `2^{b−1} > −lo − 1 = !lo`.
    1 + width(hi).max(width(!lo))
}

/// A width checked against `B_max` or 62 before it gets here.
fn narrow(bits: u32) -> u8 {
    u8::try_from(bits).unwrap_or(u8::MAX)
}
