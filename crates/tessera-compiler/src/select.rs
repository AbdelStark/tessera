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
//! RFC-0066's `BitDecompose` and `AssertEqual` take gadgets too, numbered with every other: their
//! relations are stage 1's zero-checks, with no lookup and no leaf site (RFC-0072 'Gadgets and
//! leaf sites'). Their preconditions are statement checks, `ERR-COMPILE-RANGE` when broken
//! (RFC-0039): `low + high ≤ 62`, and each input's range as RFC-0066 states it. `BitFn` is
//! virtual and takes none.
//!
//! Tables are shared by definition: a requantization table is its kind with `(b, r_by_class, lo,
//! hi)`, and two nodes asking for the same one name one table. New tables follow the lowering's,
//! numbered in order of first use, so the selection is a function of the graph (INV-IR-4).

use std::collections::BTreeMap;

use tessera_ir::{GadgetId, GadgetPolicy, Graph, Node, NodeId, Op, TableId, TensorId};
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

use crate::range::committed_source;
use crate::schedule::producers;
use crate::{CompileError, HALF_FIELD, Range, Ranges};

/// The most bits a `BitDecompose` commits, `low + high` (RFC-0066 'Integrality'): every sum of
/// its recomposition stays below `2^{62}`.
const DECOMPOSE_BITS: u32 = 62;

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
    Argmax {
        /// `Range { 29 }` for the lowest-index comparison keys (RFC-0009 Lemma 9).
        table: TableId,
    },
    /// The sampling certificate (RFC-0017).
    InverseCdf,
    /// RFC-0066's committed bits of a true integer: the booleanity, bit-padding and
    /// recomposition zero-checks of stage 1, and its committed input's padding check (RFC-0072
    /// 'The bit zero-checks'). No lookup and no leaf site.
    BitDecompose {
        /// The result bits.
        low: u8,
        /// The bits above them, 0 for none.
        high: u8,
    },
    /// RFC-0066's equality of two tensors: one zero-check of stage 1 at their real positions.
    /// No lookup and no leaf site.
    AssertEqual,
}

/// Every gadget of a graph and every table its lookups read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    /// The gadget-backed nodes in order, each with its gadget: gadget `k` is `GadgetId(k)`.
    pub gadgets: Vec<(NodeId, Gadget)>,
    /// The tables, by id: those the graph names, then gadget auxiliary tables in first-use order.
    pub tables: Vec<TableSpec>,
    reads: BTreeMap<NodeId, Vec<TableId>>,
}

impl Selection {
    /// Every lookup of `node`, in canonical order, retaining repeated table ids.
    /// Virtual top limbs and both comparison normalizers are included (RFC-0050).
    /// Returns `None` for a node without a selected gadget.
    pub fn reads(&self, node: NodeId) -> Option<&[TableId]> {
        self.reads.get(&node).map(Vec::as_slice)
    }
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
/// - [`CompileError::Range`] for a requantization key no gadget takes, a table of more than
///   `2^{B_max}` rows, or a bit operation whose precondition fails: a `BitDecompose` of more
///   than 62 bits, or of an input whose range leaves `[0, 2^{low + high})` (a committed graph
///   input's declared range must hold it instead), or an `AssertEqual` of an input outside the
///   centered field range ([`CompileError::Wrap`] for one with no bound);
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
        reads: Vec::new(),
        producers: None,
    };
    let mut gadgets = Vec::new();
    let mut reads = BTreeMap::new();
    for node in &graph.nodes {
        if let Some(gadget) = selector.node(node)? {
            gadgets.push((node.id, gadget));
            reads.insert(node.id, std::mem::take(&mut selector.reads));
        }
    }
    Ok(Selection {
        gadgets,
        tables: selector.tables,
        reads,
    })
}

struct Selector<'a> {
    graph: &'a Graph,
    ranges: &'a Ranges,
    max: u32,
    limb_bits: u8,
    tables: Vec<TableSpec>,
    reads: Vec<TableId>,
    /// Each tensor's producer, computed at the first `BitDecompose`.
    producers: Option<Vec<Option<(NodeId, u32)>>>,
}

impl Selector<'_> {
    fn node(&mut self, node: &Node) -> Result<Option<Gadget>, CompileError> {
        Ok(Some(match &node.op {
            Op::Requant { shift, lo, hi } => {
                let shifts = self.graph.shift_classes[shift.0 as usize].clone();
                self.requant(node, shifts, *lo, *hi)?
            }
            Op::TableMap { table } | Op::Bitwise { table } => {
                let table = self.sized(node, *table)?;
                self.reads.push(table);
                Gadget::Lookup { table }
            }
            Op::SoftmaxRow { exp_table, sigma } => {
                let exp = self.sized(node, *exp_table)?;
                self.reads.push(exp);
                let (reciprocal, limb, top) = self.softmax_ranges(node, *sigma)?;
                Gadget::SoftmaxRow {
                    exp,
                    reciprocal,
                    limb,
                    top,
                }
            }
            Op::RsqrtRow { .. } => {
                self.rsqrt_ranges(node)?;
                Gadget::RsqrtRow
            }
            Op::OneHot { .. } => Gadget::OneHot,
            Op::Argmax => {
                let table = self.table(node, TableKind::Range { bits: 29 })?;
                Gadget::Argmax {
                    table: self.sized(node, table)?,
                }
            }
            Op::InverseCdf { .. } => Gadget::InverseCdf,
            Op::BitDecompose { low, high } => {
                self.decomposed(node, *low, *high)?;
                Gadget::BitDecompose {
                    low: *low,
                    high: *high,
                }
            }
            Op::AssertEqual => {
                for input in 0..node.inputs.len() {
                    self.integer(node, input)?;
                }
                Gadget::AssertEqual
            }
            // Virtual, each its multilinear polynomial: no gadget (RFC-0066 '`BitFn`').
            Op::BitFn { .. } => return Ok(None),
            _ => return Ok(None),
        }))
    }

    /// RFC-0066's preconditions of a `BitDecompose { low, high }` (RFC-0072 'Gadgets and leaf
    /// sites'): `low + high ≤ 62`, and its input a true integer in `[0, 2^{low + high})`, or a
    /// committed graph input, read directly or moved by `Reshape`, `Slice`, `Transpose`,
    /// `GatherPublic` or `Expand`, whose declared range holds that interval: the bits range-check
    /// it (Lemma 1's converse), and its other readers trust its declared range.
    fn decomposed(&mut self, node: &Node, low: u8, high: u8) -> Result<(), CompileError> {
        let bits = u32::from(low) + u32::from(high);
        if low == 0 || bits > DECOMPOSE_BITS {
            return Err(CompileError::Range {
                node: node.id,
                input: 0,
                rule: "a decomposition has 1 to 62 bits: 1 ≤ low and low + high ≤ 62",
            });
        }
        let top = (1i128 << bits) - 1;
        let graph = self.graph;
        let producers = self.producers.get_or_insert_with(|| producers(graph));
        let producer = |t: TensorId| {
            producers
                .get(t.0 as usize)
                .copied()
                .flatten()
                .and_then(|(id, _)| graph.nodes.get(id.0 as usize))
        };
        let Some(&x) = node.inputs.first() else {
            return Err(CompileError::Usage {
                rule: "a BitDecompose reads its integer",
            });
        };
        if let Some(source) = committed_source(graph, producer, x) {
            let declared = self.ranges.of(source).and_then(Range::bounded);
            if !declared.is_some_and(|r| r.lo <= 0 && top <= r.hi) {
                return Err(CompileError::Range {
                    node: node.id,
                    input: 0,
                    rule: "a committed integer's declared range holds [0, 2^{low + high})",
                });
            }
            return Ok(());
        }
        self.within(node, 0, 0, top, "the integer lies in [0, 2^{low + high})")
    }

    /// Input `input` of `node` is a true integer: its range inside the centered field range, as
    /// `AssertEqual`'s two tensors must be for equality in `Fp` to be equality as integers.
    fn integer(&self, node: &Node, input: usize) -> Result<(), CompileError> {
        self.within(
            node,
            input,
            1 - HALF_FIELD,
            HALF_FIELD - 1,
            "a true integer lies in (−(P−1)/2, (P−1)/2)",
        )
    }

    /// Input `input` of `node` has a range inside `[lo, hi]`: [`CompileError::Wrap`] when it has
    /// none, [`CompileError::Range`] when it leaves the interval.
    fn within(
        &self,
        node: &Node,
        input: usize,
        lo: i128,
        hi: i128,
        rule: &'static str,
    ) -> Result<(), CompileError> {
        let range = node
            .inputs
            .get(input)
            .and_then(|&t| self.ranges.of(t))
            .and_then(Range::bounded)
            .ok_or(CompileError::Wrap {
                node: node.id,
                input,
            })?;
        if range.lo < lo || range.hi > hi {
            return Err(CompileError::Range {
                node: node.id,
                input,
                rule,
            });
        }
        Ok(())
    }

    /// RFC-0050: R's range, then every low limb and virtual top of A, then B.
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
        let mut limb = None;
        let mut top = None;
        for _ in 0..2 {
            for _ in 1..n {
                limb = Some(self.range_table(node, l)?);
            }
            top = Some(self.range_table(node, k - (n - 1) * l)?);
        }
        let top = top.ok_or(CompileError::Usage {
            rule: "both Softmax slacks have a top limb",
        })?;
        Ok((reciprocal, limb, top))
    }

    fn rsqrt_ranges(&mut self, node: &Node) -> Result<(), CompileError> {
        let refuse = || CompileError::Range {
            node: node.id,
            input: 0,
            rule: "RFC-0050's complete Rsqrt range requests fit the public policy",
        };
        if self.limb_bits == 0 || u32::from(self.limb_bits) > self.max {
            return Err(refuse());
        }
        let bound = node
            .outputs
            .first()
            .and_then(|id| self.ranges.of(*id))
            .and_then(Range::bounded)
            .ok_or_else(refuse)?;
        let high = u128::try_from(bound.hi).map_err(|_| refuse())?;
        let w = high.checked_ilog2().map_or(1, |bits| bits + 1);
        if w > 32 {
            return Err(refuse());
        }
        let r = w.div_ceil(8) - 1;
        let m = (2 * w + 55).div_ceil(8) as usize;
        let lower = root_carries(w, false, m).ok_or_else(refuse)?;
        let upper = root_carries(w, true, m).ok_or_else(refuse)?;
        self.range_table(node, w)?;
        for _ in 0..3 {
            self.range_table(node, 16)?;
        }
        self.range_table(node, 7)?;
        for _ in 0..r {
            self.range_table(node, 8)?;
        }
        self.range_table(node, w - 8 * r)?;
        let constants = vec![1; m];
        for (left, right) in [(&lower, &constants), (&constants, &upper)] {
            for carries in [left, right] {
                for _ in 0..m {
                    self.range_table(node, 8)?;
                }
                for &width in carries {
                    self.range_table(node, width)?;
                }
            }
            for _ in 0..m {
                self.range_table(node, 8)?;
            }
            for _ in 0..m {
                self.range_table(node, 1)?;
            }
        }
        Ok(())
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
        if let Some(spec) = self
            .tables
            .iter()
            .filter(|spec| spec.kind == kind)
            .min_by_key(|spec| spec.id)
        {
            self.reads.push(spec.id);
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
        self.reads.push(id);
        Ok(id)
    }
}

fn polynomial_product(a: &[u128], b: &[u128]) -> Option<Vec<u128>> {
    let mut out = vec![0u128; a.len().checked_add(b.len())?.checked_sub(1)?];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] = out[i + j].checked_add(x.checked_mul(y)?)?;
        }
    }
    Some(out)
}

// Whole weighted prefixes derive the carry widths independently of the verifier's recurrence.
fn root_carries(width: u32, increment: bool, limbs: usize) -> Option<Vec<u32>> {
    let count = width.div_ceil(8);
    let mut root: Vec<_> = (0..count)
        .map(|i| (1u128 << (width - 8 * i).min(8)) - 1)
        .collect();
    root[0] += u128::from(increment);
    let square = polynomial_product(&root, &root)?;
    let raw = polynomial_product(&square, &[65535, 0, 65535, 0, 65535, 0, 127])?;
    if raw.len() > limbs {
        return None;
    }
    let (mut prefix, mut previous_box) = (0u128, 0u128);
    let mut widths = Vec::with_capacity(limbs);
    for j in 0..limbs {
        let a = raw.get(j).copied().unwrap_or(0);
        prefix = prefix.checked_add(a.checked_mul(1u128.checked_shl(8 * j as u32)?)?)?;
        let carry = prefix >> (8 * (j + 1));
        let bits = (u128::BITS - carry.leading_zeros()).max(1);
        let full_box = (1u128 << bits) - 1;
        if a + previous_box >= 1 << 40 || 255 + 256 * full_box >= 1 << 40 {
            return None;
        }
        previous_box = full_box;
        widths.push(bits);
    }
    (prefix >> (8 * limbs) == 0).then_some(widths)
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
