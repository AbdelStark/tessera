// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gadget selection and table sizing (RFC-0006 'Gadget selection', RFC-0009).
//!
//! Every gadget-backed node gets its gadget, numbered in node order. A `Requant` takes one of two
//! by its key's signed width `b`, from range analysis, and its `n` shift classes, whose index
//! takes `⌈log2 n⌉` more key bits in a table:
//!
//! - **`Requant1`** reads `Requant { b, r_by_class, lo, hi }` once, when `b + ⌈log2 n⌉ ≤ B_max`.
//! - **`RequantSplit`** sends the low `s = b + ⌈log2 n⌉ − B_max` bits of the offset key to
//!   `Range { s }` and `⌊z / 2^s⌋` to `Requant { b − s, r_by_class − s, lo, hi }`. The nested floor
//!   is the requantization only when `s ≤ r_c − 1` for every class, the halves are true integers
//!   only when `b ≤ 62` (RFC-0009 Lemma 2), and the low part's table must fit too: `s ≤ B_max`.
//!
//! Any other key is `ERR-COMPILE-RANGE`. A `TableMap`, `Bitwise` or `SoftmaxRow` reads a table of
//! its own, which must hold at most `2^{B_max}` rows, since those lookups have no split.
//!
//! Tables are shared by definition: a requantization table is its kind with `(b, r_by_class, lo,
//! hi)`, and two nodes asking for the same one name one table. New tables follow the lowering's,
//! numbered in order of first use, so the selection is a function of the graph (INV-IR-4).

use tessera_ir::{GadgetId, GadgetPolicy, Graph, Node, NodeId, Op, TableId};
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

use crate::{CompileError, Range, Ranges};

/// The bound of RFC-0009 Lemma 2 on a split key: both halves stay true integers.
const SPLIT_KEY_BITS: u32 = 62;

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
    /// One lookup into the node's table: `TableMap` and `Bitwise`.
    Lookup {
        /// The table.
        table: TableId,
    },
    /// The max certificate and reciprocal bracket, with its `Exp` table.
    SoftmaxRow {
        /// The `Exp` table.
        exp: TableId,
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
    /// The tables, by id: those the graph names, then the requantizations'.
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
            Op::SoftmaxRow { exp_table, .. } => Gadget::SoftmaxRow {
                exp: self.sized(node, *exp_table)?,
            },
            Op::RsqrtRow { .. } => Gadget::RsqrtRow,
            Op::OneHot { .. } => Gadget::OneHot,
            Op::Argmax => Gadget::Argmax,
            Op::InverseCdf { .. } => Gadget::InverseCdf,
            _ => return Ok(None),
        }))
    }

    /// `Requant1` or `RequantSplit` for the key of `node`, by its width.
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
        let split = b + class_bits - self.max;
        if split > self.max {
            return Err(refuse(
                "a split key's low part fits a table of 2^{B_max} rows",
            ));
        }
        let least = shifts.iter().copied().min().unwrap_or(0);
        if split >= u32::from(least) {
            return Err(refuse(
                "a split key keeps its rounding: s ≤ r − 1 for every class",
            ));
        }
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
