// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Range analysis (RFC-0006 'Range analysis'): an interval for every tensor of a graph, from
//! public data alone, and the true-integer checks at every gadget input.
//!
//! - **Inputs** keep the ranges the graph declares for them: tokens in `[0, vocab)`, positions,
//!   weights in `[−127, 127]`, activations over their type. The compiler declares them from the
//!   claim list's contents, so the verifier re-derives them (INV-IR-5, SP-7).
//! - **Linear and degree-2 operations** take exact `i128` interval arithmetic. `MatMulWeight` uses
//!   only the public bound `max|W| · fan_in · max|x|`, never a weight's value: a bound derived from
//!   secret weights would leak them through the claim list. An interval that would leave `i128`
//!   is [`Range::Unbounded`]; a virtual tensor may be, since it is computed modulo `P`.
//! - **Gadget outputs** take the range their gadget's lookups enforce (RFC-0009).
//! - **The relational fact.** `SoftmaxRow` emits `R_q · s_q ≤ 2^{2σ}` (RFC-0009 Lemma 6), with
//!   `s_q = Σ_i e_qi`. When the attention output multiplies `PV = MatMulAct(e, v)` by that row's
//!   `R`, as RFC-0006's lowering does, `R_q · Σ_i e_qi v_ic` lies between `2^{2σ} · min(v)` and
//!   `2^{2σ} · max(v)`, with 0 in between: `e ≥ 0`, `R ≥ 0`. The product of the operands' bounds
//!   would be about `2^σ` times the context length wider. The fact holds for honest values, whose
//!   exponentials are zero at masked positions: the reference softmax has no masked term, and
//!   #279 makes the protocol enforce it.
//! - **True integers.** Every gadget input that must be an integer needs a bounded range inside
//!   `(−(P−1)/2, (P−1)/2)`, and inside what its gadget or table accepts. Without a bound it is
//!   `ERR-COMPILE-WRAP`, outside one `ERR-COMPILE-RANGE`. Inside the gadgets the field value of
//!   such an input is then its true integer, whatever wraps happened upstream: it is an integer
//!   polynomial of values whose own exact range is bounded (RFC-0006 'True-integer rule').
//!
//! The analysis reads the graph and the tables' specifications, never a weight, so ranges cannot
//! depend on weight values (INV-CL-1), and two runs on the same inputs agree (INV-IR-4).

use tessera_ir::{Graph, IntRange, Node, NodeId, Op, TableId, TensorId};
use tessera_tables::{TableKind, TableSpec};

use crate::CompileError;

/// `(P − 1)/2` for the Goldilocks prime `P = 2^64 − 2^32 + 1`: a true integer lies strictly
/// within it on either side (RFC-0006 'True-integer rule').
pub const HALF_FIELD: i128 = (1 << 63) - (1 << 31);

/// A `SoftmaxRow` score's bound: `|x| < 2^21` (RFC-0009 'SoftmaxRow').
const SCORE_BOUND: i128 = 1 << 21;

/// An `RsqrtRow` sum's bound with its epsilon: `u + e < 2^55` (RFC-0009 'RsqrtRow').
const RSQRT_BOUND: i128 = 1 << 55;

/// The values a tensor can take, as range analysis bounds them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Range {
    /// Every value lies in the interval.
    Bounded(IntRange),
    /// The interval would leave `i128`: no bound is claimed.
    Unbounded,
}

impl Range {
    /// The interval `[lo, hi]`.
    pub const fn new(lo: i128, hi: i128) -> Self {
        Self::Bounded(IntRange { lo, hi })
    }

    /// The interval, if bounded.
    pub fn bounded(self) -> Option<IntRange> {
        match self {
            Self::Bounded(range) => Some(range),
            Self::Unbounded => None,
        }
    }
}

/// The range of every tensor of a graph, by id.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ranges(Vec<Range>);

impl Ranges {
    /// The range of `tensor`, or `None` for a tensor the graph does not hold.
    pub fn of(&self, tensor: TensorId) -> Option<Range> {
        self.0.get(usize::try_from(tensor.0).ok()?).copied()
    }

    /// Every range, by tensor id.
    pub fn all(&self) -> &[Range] {
        &self.0
    }
}

/// The range of every tensor of `graph`, whose lookups name tables among `tables`.
///
/// # Errors
///
/// The first failure in node order:
///
/// - [`CompileError::Wrap`] for a gadget input that must be an integer and has no bound;
/// - [`CompileError::Range`] for one whose bound leaves the centered field range or what its
///   gadget or table accepts;
/// - [`CompileError::Table`] for a table missing from `tables` or of the wrong kind;
/// - [`CompileError::Graph`] for a graph that breaks a rule of RFC-0006.
pub fn analyze(graph: &Graph, tables: &[TableSpec]) -> Result<Ranges, CompileError> {
    graph.validate()?;
    let mut analysis = Analysis {
        graph,
        tables,
        ranges: graph
            .tensors
            .iter()
            .map(|t| Range::Bounded(t.range))
            .collect(),
        producers: vec![None; graph.tensors.len()],
    };
    for node in &graph.nodes {
        let outputs = analysis.node(node)?;
        for (&tensor, range) in node.outputs.iter().zip(outputs) {
            analysis.ranges[index(tensor)] = range;
            analysis.producers[index(tensor)] = Some(node.id);
        }
    }
    Ok(Ranges(analysis.ranges))
}

struct Analysis<'a> {
    graph: &'a Graph,
    tables: &'a [TableSpec],
    /// Inputs' declared ranges, then each output's as its node is reached.
    ranges: Vec<Range>,
    /// The node that produced each tensor so far.
    producers: Vec<Option<NodeId>>,
}

impl Analysis<'_> {
    fn range(&self, tensor: TensorId) -> Range {
        self.ranges[index(tensor)]
    }

    fn shape(&self, tensor: TensorId) -> &[u32] {
        &self.graph.tensors[index(tensor)].shape
    }

    fn producer(&self, tensor: TensorId) -> Option<&Node> {
        self.producers[index(tensor)].map(|id| &self.graph.nodes[id.0 as usize])
    }

    fn table(&self, node: &Node, table: TableId) -> Result<&TableKind, CompileError> {
        self.tables
            .iter()
            .find(|spec| spec.id == table)
            .map(|spec| &spec.kind)
            .ok_or(CompileError::Table {
                node: node.id,
                table,
                rule: "no specification names it",
            })
    }

    /// The ranges of `node`'s outputs, after checking its gadget inputs.
    fn node(&self, node: &Node) -> Result<Vec<Range>, CompileError> {
        let input = |k: usize| self.range(node.inputs[k]);
        let bounded = |k: usize, f: &dyn Fn(IntRange) -> Option<IntRange>| {
            Ok(vec![lift(input(k).bounded().and_then(f))])
        };
        match &node.op {
            Op::MatMulWeight { .. } => {
                let fan_in = u64::from(self.shape(node.inputs[1])[1]);
                let weight = input(1).bounded().and_then(magnitude);
                let bound = input(0)
                    .bounded()
                    .and_then(magnitude)
                    .zip(weight)
                    .and_then(|(x, w)| x.checked_mul(w)?.checked_mul(i128::from(fan_in)));
                Ok(vec![lift(bound.map(symmetric))])
            }
            Op::Add | Op::Sub => {
                let sum = input(0)
                    .bounded()
                    .zip(input(1).bounded())
                    .and_then(|(a, b)| {
                        if node.op == Op::Add {
                            add(a, b)
                        } else {
                            add(a, negate(b)?)
                        }
                    });
                Ok(vec![lift(sum)])
            }
            Op::ConstMul { c } => {
                let (low, high) = extremes(&self.graph.constants[c.0 as usize]);
                bounded(0, &|x| hull(mul(x, point(low))?, mul(x, point(high))?))
            }
            Op::ConstAdd { c } => {
                let (low, high) = extremes(&self.graph.constants[c.0 as usize]);
                bounded(0, &|x| add(x, IntRange { lo: low, hi: high }))
            }
            Op::Rotate { rope } => {
                let TableKind::Rope { bits, .. } = self.table(node, *rope)? else {
                    return Err(CompileError::Table {
                        node: node.id,
                        table: *rope,
                        rule: "Rotate takes a Rope table",
                    });
                };
                // Each output is a·C − b·S or a·S + b·C with |C|, |S| ≤ 2^bits: at most
                // 2^{bits+1} times the largest input magnitude.
                let gain = 1i128 << (u32::from(*bits) + 1);
                bounded(0, &|x| Some(symmetric(magnitude(x)?.checked_mul(gain)?)))
            }
            Op::Sum { axis } => {
                let n = i128::from(self.shape(node.inputs[0])[usize::from(*axis)]);
                bounded(0, &|x| mul(x, point(n)))
            }
            Op::SumSquares { axis } => {
                let n = i128::from(self.shape(node.inputs[0])[usize::from(*axis)]);
                bounded(0, &|x| mul(square(x)?, point(n)))
            }
            Op::Reshape | Op::Slice { .. } | Op::GatherPublic { .. } => Ok(vec![input(0)]),
            Op::MatMulAct { .. } => {
                let inner = i128::from(*self.shape(node.inputs[0]).last().unwrap_or(&1));
                let product = input(0)
                    .bounded()
                    .zip(input(1).bounded())
                    .and_then(|(a, b)| mul(mul(a, b)?, point(inner)));
                Ok(vec![lift(product)])
            }
            Op::Mul => {
                if let Some(range) = self.softmax_fact(node) {
                    return Ok(vec![range]);
                }
                let product = input(0)
                    .bounded()
                    .zip(input(1).bounded())
                    .and_then(|(a, b)| mul(a, b));
                Ok(vec![lift(product)])
            }
            Op::OneHot { vocab } => {
                self.within(
                    node,
                    0,
                    0,
                    i128::from(*vocab) - 1,
                    "a token lies in [0, vocab)",
                )?;
                Ok(vec![Range::new(0, 1)])
            }
            Op::Requant { shift, lo, hi } => {
                self.integer(node, 0)?;
                if node.inputs.len() > 1 {
                    let classes = self.graph.shift_classes[shift.0 as usize].len();
                    let last = i128::try_from(classes).unwrap_or(i128::MAX) - 1;
                    self.within(
                        node,
                        1,
                        0,
                        last,
                        "a channel class indexes the shift classes",
                    )?;
                }
                Ok(vec![Range::new(i128::from(*lo), i128::from(*hi))])
            }
            Op::TableMap { table } => {
                let kind = self.table(node, *table)?;
                let (lo, hi) = keys(kind).ok_or(CompileError::Table {
                    node: node.id,
                    table: *table,
                    rule: "TableMap takes a table of single keys",
                })?;
                self.within(node, 0, lo, hi, "a key lies in its table's keys")?;
                Ok(vec![values(kind)])
            }
            Op::Bitwise { table } => {
                let kind = self.table(node, *table)?;
                if !matches!(kind, TableKind::Bitwise8 { .. }) {
                    return Err(CompileError::Table {
                        node: node.id,
                        table: *table,
                        rule: "Bitwise takes a Bitwise8 table",
                    });
                }
                for k in 0..2 {
                    self.within(node, k, 0, 255, "a byte operand lies in [0, 256)")?;
                }
                Ok(vec![values(kind)])
            }
            Op::SoftmaxRow { exp_table, sigma } => self.softmax(node, *exp_table, *sigma),
            Op::RsqrtRow { kappa, eps_int } => {
                let eps = i128::from(*eps_int);
                self.within(node, 0, 0, RSQRT_BOUND - 1 - eps, "u + e lies in [0, 2^55)")?;
                Ok(vec![Range::new(0, (1 << (u32::from(*kappa) + 1)) - 1)])
            }
            Op::Argmax | Op::InverseCdf { .. } => {
                self.integer(node, 0)?;
                self.within(node, 1, 0, 1, "a one-hot entry is 0 or 1")?;
                Ok(Vec::new())
            }
        }
    }

    /// `SoftmaxRow`'s outputs: the exponentials in its `Exp` table's values, and `R_q`, which its
    /// range check bounds by `2^{σ+1}` (RFC-0009). The scores are true integers with `|x| < 2^21`,
    /// and their spread fits the table's keys, since each key is the row maximum less a score.
    fn softmax(&self, node: &Node, table: TableId, sigma: u8) -> Result<Vec<Range>, CompileError> {
        let kind = self.table(node, table)?;
        let TableKind::Exp {
            key_bits,
            sigma: table_sigma,
            ..
        } = *kind
        else {
            return Err(CompileError::Table {
                node: node.id,
                table,
                rule: "SoftmaxRow takes an Exp table",
            });
        };
        if table_sigma != sigma {
            return Err(CompileError::Table {
                node: node.id,
                table,
                rule: "SoftmaxRow's σ is its Exp table's",
            });
        }
        let scores = self.within(
            node,
            0,
            1 - SCORE_BOUND,
            SCORE_BOUND - 1,
            "a score lies in (−2^21, 2^21)",
        )?;
        if scores.hi - scores.lo >= 1 << key_bits {
            return Err(CompileError::Range {
                node: node.id,
                input: 0,
                rule: "the scores' spread fits the Exp table's keys",
            });
        }
        Ok(vec![
            values(kind),
            Range::new(0, (1 << (u32::from(sigma) + 1)) - 1),
        ])
    }

    /// The bound `R_q · Σ_i e_qi v_ic` takes when `node` is `Mul(MatMulAct(e, v), R)` with `e` and
    /// `R` the outputs of one `SoftmaxRow`: `2^{2σ}` times `v`'s range widened to 0 (module
    /// documentation).
    fn softmax_fact(&self, node: &Node) -> Option<Range> {
        let &[product, reciprocal] = node.inputs.as_slice() else {
            return None;
        };
        let pv = self.producer(product)?;
        let &[e, v] = pv.inputs.as_slice() else {
            return None;
        };
        if pv.op
            != (Op::MatMulAct {
                transpose_rhs: false,
            })
        {
            return None;
        }
        let softmax = self.producer(e)?;
        let Op::SoftmaxRow { sigma, .. } = softmax.op else {
            return None;
        };
        if softmax.outputs.as_slice() != [e, reciprocal] {
            return None;
        }
        let v = self.range(v).bounded()?;
        let scale = 1i128 << (2 * u32::from(sigma));
        Some(Range::new(
            v.lo.min(0).checked_mul(scale)?,
            v.hi.max(0).checked_mul(scale)?,
        ))
    }

    /// Input `k` of `node` as a true integer: bounded, inside the centered field range.
    fn integer(&self, node: &Node, k: usize) -> Result<IntRange, CompileError> {
        let rule = "a true integer lies in (−(P−1)/2, (P−1)/2)";
        self.within(node, k, 1 - HALF_FIELD, HALF_FIELD - 1, rule)
    }

    /// Input `k` of `node` as a true integer inside `[lo, hi]`: every gadget's and table's
    /// domain lies inside the centered field range, so a bound inside it is a true integer.
    fn within(
        &self,
        node: &Node,
        k: usize,
        lo: i128,
        hi: i128,
        rule: &'static str,
    ) -> Result<IntRange, CompileError> {
        let Some(range) = self.range(node.inputs[k]).bounded() else {
            return Err(CompileError::Wrap {
                node: node.id,
                input: k,
            });
        };
        if range.lo < lo || range.hi > hi {
            return Err(CompileError::Range {
                node: node.id,
                input: k,
                rule,
            });
        }
        Ok(range)
    }
}

/// The keys a single-input lookup into a table of `kind` may take, as an interval; `None` for the
/// kinds no single key reaches.
fn keys(kind: &TableKind) -> Option<(i128, i128)> {
    let below = |bits: u8| (1i128 << bits) - 1;
    let centered = |b: u8| (-(1i128 << (b - 1)), (1i128 << (b - 1)) - 1);
    match *kind {
        TableKind::Range { bits } => Some((0, below(bits))),
        TableKind::Requant { b, .. } | TableKind::Silu { b, .. } => Some(centered(b)),
        TableKind::Pow2 { max_exp } => Some((0, i128::from(max_exp) - 1)),
        TableKind::AesSbox | TableKind::Gf256Mul { .. } => Some((0, 255)),
        TableKind::Exp { key_bits, .. } => Some((0, below(key_bits))),
        TableKind::Exp2Frac { frac_bits, .. } => Some((0, below(frac_bits))),
        TableKind::Bitwise8 { .. } | TableKind::Rope { .. } => None,
    }
}

/// The values a table of `kind` holds, as an interval (RFC-0008 'Table specifications').
fn values(kind: &TableKind) -> Range {
    match *kind {
        TableKind::Range { .. } | TableKind::Rope { .. } => Range::new(0, 0),
        TableKind::Requant { lo, hi, .. } => Range::new(i128::from(lo), i128::from(hi)),
        TableKind::Pow2 { max_exp } => Range::new(1, 1 << (u32::from(max_exp) - 1)),
        TableKind::Bitwise8 { .. } | TableKind::AesSbox | TableKind::Gf256Mul { .. } => {
            Range::new(0, 255)
        }
        TableKind::Exp { sigma, .. } => Range::new(0, 1 << u32::from(sigma)),
        // `exp_pos` stays at or below 2^63 in Q62, so the value at or below 2^{out_bits + 1}.
        TableKind::Exp2Frac { out_bits, .. } => Range::new(0, 1 << (u32::from(out_bits) + 1)),
        TableKind::Silu { .. } => Range::new(-128, 127),
    }
}

fn index(tensor: TensorId) -> usize {
    tensor.0 as usize
}

fn lift(range: Option<IntRange>) -> Range {
    range.map_or(Range::Unbounded, Range::Bounded)
}

fn point(x: i128) -> IntRange {
    IntRange { lo: x, hi: x }
}

fn symmetric(bound: i128) -> IntRange {
    IntRange {
        lo: -bound,
        hi: bound,
    }
}

/// The largest magnitude in the interval, if it fits.
fn magnitude(x: IntRange) -> Option<i128> {
    Some(x.lo.checked_abs()?.max(x.hi.checked_abs()?))
}

fn add(a: IntRange, b: IntRange) -> Option<IntRange> {
    Some(IntRange {
        lo: a.lo.checked_add(b.lo)?,
        hi: a.hi.checked_add(b.hi)?,
    })
}

fn negate(x: IntRange) -> Option<IntRange> {
    Some(IntRange {
        lo: x.hi.checked_neg()?,
        hi: x.lo.checked_neg()?,
    })
}

/// The product's interval: a bilinear function's extremes lie at the corners.
fn mul(a: IntRange, b: IntRange) -> Option<IntRange> {
    let corners = [
        a.lo.checked_mul(b.lo)?,
        a.lo.checked_mul(b.hi)?,
        a.hi.checked_mul(b.lo)?,
        a.hi.checked_mul(b.hi)?,
    ];
    Some(IntRange {
        lo: *corners.iter().min()?,
        hi: *corners.iter().max()?,
    })
}

/// The squares of the interval's values: 0 at least when it holds 0.
fn square(x: IntRange) -> Option<IntRange> {
    let top = magnitude(x)?.checked_mul(magnitude(x)?)?;
    let bottom = if x.lo <= 0 && x.hi >= 0 {
        0
    } else {
        let near = x.lo.checked_abs()?.min(x.hi.checked_abs()?);
        near.checked_mul(near)?
    };
    Some(IntRange {
        lo: bottom,
        hi: top,
    })
}

fn hull(a: IntRange, b: IntRange) -> Option<IntRange> {
    Some(IntRange {
        lo: a.lo.min(b.lo),
        hi: a.hi.max(b.hi),
    })
}

/// A constant's least and greatest value: a scalar is both.
fn extremes(values: &[i64]) -> (i128, i128) {
    let lo = values.iter().copied().min().unwrap_or(0);
    let hi = values.iter().copied().max().unwrap_or(0);
    (i128::from(lo), i128::from(hi))
}
