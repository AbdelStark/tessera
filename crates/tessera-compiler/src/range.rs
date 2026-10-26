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
//! - **Relational facts.** `SoftmaxRow` emits `R_q · s_q ≤ 2^{2σ}` (RFC-0009 Lemma 6), with
//!   `s_q = Σ_i e_qi`. When the attention output multiplies `PV = MatMulAct(e, v)` by that row's
//!   `R`, as RFC-0006's lowering does, `R_q · Σ_i e_qi v_ic` lies between `2^{2σ} · min(v)` and
//!   `2^{2σ} · max(v)`, with 0 in between: `e ≥ 0`, `R ≥ 0`. The product of the operands' bounds
//!   would be about `2^σ` times the context length wider. The fact sums every key position, and
//!   holds because the exponentials are zero at masked positions: `SoftmaxRow` looks a masked
//!   position up at its table's last key, which the table maps to 0 (RFC-0030). `OneHot` emits that its vectors have exactly one 1
//!   (RFC-0009 Lemma 8), so `MatMulWeight` on them selects a column of the weight and keeps the
//!   weight's range: the embedding rows stay INT8 rather than growing with the vocabulary.
//! - **Committed graph inputs.** A graph input committed in the advice holds whatever the prover
//!   commits, so every element of it must be read by an operation that proves its declared range:
//!   `OneHot` (`[0, vocab)`), reading it directly, or `BitDecompose` (`[0, 2^{low + high})`),
//!   directly or through moves. [`analyze`] refuses a graph that leaves one unproven;
//!   [`analyze_fragment`] does not, for a circuit whose committed inputs a larger graph proves
//!   (RFC-0006 'Committed graph inputs').
//! - **True integers.** Every gadget input that must be an integer needs a bounded range inside
//!   `(−(P−1)/2, (P−1)/2)`, and inside what its gadget or table accepts. Without a bound it is
//!   `ERR-COMPILE-WRAP`, outside one `ERR-COMPILE-RANGE`. Inside the gadgets the field value of
//!   such an input is then its true integer, whatever wraps happened upstream: it is an integer
//!   polynomial of values whose own exact range is bounded (RFC-0006 'True-integer rule').
//!
//! The analysis reads the graph and the tables' specifications, never a weight, so ranges cannot
//! depend on weight values (INV-CL-1), and two runs on the same inputs agree (INV-IR-4).

use tessera_ir::{CommitmentClass, Graph, IntRange, Node, NodeId, NodeKind, Op, TableId, TensorId};
use tessera_tables::{TableKind, TableSpec};

use crate::CompileError;

/// The elements of a tensor that an operation reads: every one, or those whose flat index,
/// outermost axis first, is set.
#[derive(Clone, Debug)]
enum Reached {
    All,
    Some(Vec<bool>),
}

impl Reached {
    fn complete(&self) -> bool {
        match self {
            Self::All => true,
            Self::Some(bits) => bits.iter().all(|&b| b),
        }
    }

    fn merge(into: &mut Option<Self>, other: Self) {
        *into = Some(match (into.take(), other) {
            (None, x) => x,
            (Some(Self::All), _) | (Some(_), Self::All) => Self::All,
            (Some(Self::Some(mut a)), Self::Some(b)) => {
                a.iter_mut().zip(b).for_each(|(a, b)| *a |= b);
                Self::Some(a)
            }
        });
    }

    /// The elements of a move's input, of shape `from`, that its output's elements in `self`, of
    /// shape `to`, hold. `source(o, i)` writes into `i` the input index whose value output index
    /// `o` holds. When `onto`, the move reads every input element, so all of its output holds all
    /// of its input. `None`, read as nothing proven, when an index cannot be formed.
    fn back(
        self,
        from: &[u32],
        to: &[u32],
        onto: bool,
        source: impl Fn(&[usize], &mut [usize]) -> Option<()>,
    ) -> Option<Self> {
        if onto && matches!(self, Self::All) {
            return Some(Self::All);
        }
        let (n_from, n_to) = (elements(from)?, elements(to)?);
        let mut bits = vec![false; n_from];
        let (mut o, mut i) = (vec![0usize; to.len()], vec![0usize; from.len()]);
        for flat in 0..n_to {
            if matches!(&self, Self::Some(read) if !read[flat]) {
                advance(&mut o, to);
                continue;
            }
            source(&o, &mut i)?;
            let at = i.iter().zip(from).try_fold(0usize, |at, (&k, &d)| {
                at.checked_mul(d as usize)?.checked_add(k)
            })?;
            *bits.get_mut(at)? = true;
            advance(&mut o, to);
        }
        Some(Self::Some(bits))
    }
}

/// The number of elements of a tensor of `shape`, if it fits a `usize`.
fn elements(shape: &[u32]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
}

/// The next index of a tensor of `shape` in row-major order, the last axis fastest.
fn advance(index: &mut [usize], shape: &[u32]) {
    for (k, &d) in index.iter_mut().zip(shape).rev() {
        *k += 1;
        if *k < d as usize {
            return;
        }
        *k = 0;
    }
}

/// `(P − 1)/2` for the Goldilocks prime `P = 2^64 − 2^32 + 1`: a true integer lies strictly
/// within it on either side (RFC-0006 'True-integer rule').
pub const HALF_FIELD: i128 = (1 << 63) - (1 << 31);

/// A `SoftmaxRow` score's bound: `|x| ≤ 2^21` (RFC-0009 'SoftmaxRow'), attained by INT8 queries
/// and keys of width 128 at `−128`.
const SCORE_BOUND: i128 = 1 << 21;

/// An `RsqrtRow` sum's bound with its epsilon: `u + e < 2^55` (RFC-0009 'RsqrtRow'). Its root takes
/// the range its check enforces, `[0, 2^w)` with `w` from `d`, `κ` and `e` (RFC-0022).
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
/// - [`CompileError::Graph`] for a graph that breaks a rule of RFC-0006;
///
/// then, after every node, [`CompileError::Unproven`] for the first graph input committed in the
/// advice with an element no operation proves in its declared range.
pub fn analyze(graph: &Graph, tables: &[TableSpec]) -> Result<Ranges, CompileError> {
    let analysis = Analysis::run(graph, tables)?;
    analysis.prove_inputs()?;
    Ok(Ranges(analysis.ranges))
}

/// The ranges of a fragment: a circuit whose committed inputs a larger graph proves, such as
/// AES-128-CTR alone, whose key and plaintext it does not prove bytes (RFC-0006 'Committed graph
/// inputs').
///
/// It checks what [`analyze`] checks but that rule: a committed input's declared range is taken
/// as given. A proof's graph goes through [`analyze`].
///
/// # Errors
///
/// As [`analyze`], but never [`CompileError::Unproven`].
pub fn analyze_fragment(graph: &Graph, tables: &[TableSpec]) -> Result<Ranges, CompileError> {
    Ok(Ranges(Analysis::run(graph, tables)?.ranges))
}

struct Analysis<'a> {
    graph: &'a Graph,
    tables: &'a [TableSpec],
    /// Inputs' declared ranges, then each output's as its node is reached.
    ranges: Vec<Range>,
    /// The node that produced each tensor so far.
    producers: Vec<Option<NodeId>>,
}

impl<'a> Analysis<'a> {
    /// Every node's ranges, in node order.
    fn run(graph: &'a Graph, tables: &'a [TableSpec]) -> Result<Self, CompileError> {
        graph.validate()?;
        let mut analysis = Self {
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
        Ok(analysis)
    }

    /// RFC-0006 'Committed graph inputs': every element of each graph input committed in the
    /// advice is read by an operation whose relations prove it lies in the input's declared
    /// range, and pin its padded positions.
    ///
    /// - `BitDecompose { low, high }` proves `[0, 2^{low + high})` (RFC-0066 Lemma 1's converse),
    ///   when the declared range holds it, read directly or moved by `Reshape`, `Slice`,
    ///   `Transpose`, `GatherPublic` or `Expand`: RFC-0066's padding check pins the input's padded
    ///   positions. [`Self::node`] refuses a declared range that does not hold it; the condition
    ///   is checked again here, so that no reader proves more than it does.
    /// - `OneHot { vocab }` proves `[0, vocab)` (RFC-0009 Lemma 8), when the declared range holds
    ///   `[0, vocab − 1]`, read directly: its row selector pins its own input's padded rows, which
    ///   through a move are not the input's.
    ///
    /// No other operation is recognized: a `Bitwise` key `a + 2^8·b` binds neither operand
    /// without the other's range, and a `TableMap` would need RFC-0066's padding check on what it
    /// reads. An input committed in the weight commitment takes the domain `R_W`'s publisher
    /// vouches for (RFC-0048) and needs no reader.
    fn prove_inputs(&self) -> Result<(), CompileError> {
        let mut proven = vec![None; self.graph.tensors.len()];
        for node in &self.graph.nodes {
            let Some(&input) = node.inputs.first() else {
                continue;
            };
            // The walk the node's own checks take, before any element is counted.
            let Some(source) = self.committed_source(input) else {
                continue;
            };
            let declared = self.range(source).bounded();
            let holds = |top: i128| declared.is_some_and(|r| r.lo <= 0 && top <= r.hi);
            let proves = match node.op {
                Op::BitDecompose { low, high } => {
                    holds((1i128 << (u32::from(low) + u32::from(high))) - 1)
                }
                Op::OneHot { vocab } => input == source && holds(i128::from(vocab) - 1),
                _ => false,
            };
            if let Some((reached_source, reached)) = proves.then(|| self.reached(input)).flatten() {
                // Both walks take the same moves, so they end at the same input.
                if reached_source == source {
                    Reached::merge(&mut proven[index(source)], reached);
                }
            }
        }
        let advice = NodeKind::Committed {
            commitment: CommitmentClass::Advice,
        };
        for spec in &self.graph.tensors {
            let input = spec.kind == advice && self.producer(spec.id).is_none();
            if input
                && !proven[index(spec.id)]
                    .as_ref()
                    .is_some_and(Reached::complete)
            {
                return Err(CompileError::Unproven {
                    tensor: spec.id,
                    rule: "every element of a committed graph input is read by an operation that \
                           proves its declared range",
                });
            }
        }
        Ok(())
    }

    /// The committed graph input whose values `tensor` holds, moved by `Reshape`, `Slice`,
    /// `Transpose`, `GatherPublic` or `Expand`, with the elements of it that `tensor` holds. `None`
    /// when `tensor` holds no committed input's values unchanged, or when the elements cannot be
    /// counted in a `usize`.
    fn reached(&self, mut tensor: TensorId) -> Option<(TensorId, Reached)> {
        let mut reached = Reached::All;
        while let Some(node) = self.producer(tensor) {
            let input = node.inputs[0];
            let (from, to) = (self.shape(input), self.shape(tensor));
            reached = match &node.op {
                Op::Reshape => reached,
                // Each moves an element of the output's index `o` from the input's index below,
                // and the first two read every input element.
                Op::Transpose { perm } => reached.back(from, to, true, |o, i| {
                    for (axis, &p) in perm.iter().enumerate() {
                        *i.get_mut(usize::from(p))? = o[axis];
                    }
                    Some(())
                })?,
                Op::Expand { axis, .. } => {
                    let axis = usize::from(*axis);
                    reached.back(from, to, true, |o, i| {
                        i[..axis].copy_from_slice(o.get(..axis)?);
                        i[axis..].copy_from_slice(o.get(axis + 1..)?);
                        Some(())
                    })?
                }
                Op::Slice { axis, start, .. } => {
                    let (axis, start) = (usize::from(*axis), *start as usize);
                    reached.back(from, to, false, |o, i| {
                        i.copy_from_slice(o);
                        *i.get_mut(axis)? = o.get(axis)?.checked_add(start)?;
                        Some(())
                    })?
                }
                Op::GatherPublic { index } => {
                    let rows = self.graph.constants.get(index.0 as usize)?;
                    reached.back(from, to, false, |o, i| {
                        i.copy_from_slice(o);
                        i[0] = usize::try_from(*rows.get(o[0])?).ok()?;
                        Some(())
                    })?
                }
                _ => return None,
            };
            tensor = input;
        }
        let kind = self.graph.tensors.get(index(tensor)).map(|t| t.kind);
        matches!(kind, Some(NodeKind::Committed { .. })).then_some((tensor, reached))
    }

    fn range(&self, tensor: TensorId) -> Range {
        self.ranges[index(tensor)]
    }

    fn shape(&self, tensor: TensorId) -> &[u32] {
        &self.graph.tensors[index(tensor)].shape
    }

    fn producer(&self, tensor: TensorId) -> Option<&Node> {
        self.producers[index(tensor)].map(|id| &self.graph.nodes[id.0 as usize])
    }

    /// The committed graph input whose values `tensor` holds, moved but unchanged by `Reshape`,
    /// `Slice`, `Transpose`, `GatherPublic` or `Expand`, if there is one. Its range is then that
    /// input's declared range, which nothing has proven (RFC-0066 'Ranges').
    fn committed_source(&self, tensor: TensorId) -> Option<TensorId> {
        committed_source(self.graph, |t| self.producer(t), tensor)
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
                if self
                    .producer(node.inputs[0])
                    .is_some_and(|p| matches!(p.op, Op::OneHot { .. }))
                {
                    // A one-hot vector selects one column: the weight's own range.
                    return Ok(vec![input(1)]);
                }
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
            Op::Reshape
            | Op::Slice { .. }
            | Op::GatherPublic { .. }
            | Op::Transpose { .. }
            | Op::Expand { .. } => Ok(vec![input(0)]),
            Op::MatMulAct { .. } => {
                let inner = i128::from(*self.shape(node.inputs[0]).last().unwrap_or(&1));
                let product = input(0)
                    .bounded()
                    .zip(input(1).bounded())
                    .and_then(|(a, b)| mul(mul(a, b)?, point(inner)));
                Ok(vec![lift(product)])
            }
            Op::Mul { .. } => {
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
                let classes = classes(kind);
                if (classes > 1) != (node.inputs.len() > 1) {
                    return Err(CompileError::Table {
                        node: node.id,
                        table: *table,
                        rule: "the channel classes come with a table of several classes, and only \
                               with one",
                    });
                }
                self.within(node, 0, lo, hi, "a key lies in its table's keys")?;
                if classes > 1 {
                    self.within(
                        node,
                        1,
                        0,
                        classes - 1,
                        "a channel class indexes the table's classes",
                    )?;
                }
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
            Op::RsqrtRow {
                kappa,
                eps_int,
                dim,
            } => {
                let eps = i128::from(*eps_int);
                self.within(node, 0, 0, RSQRT_BOUND - 1 - eps, "u + e lies in [0, 2^55)")?;
                let bits = root_bits(*kappa, *eps_int, *dim).ok_or(CompileError::Range {
                    node: node.id,
                    input: 0,
                    rule: "the roots' range check covers every row: e ≥ 1, and w ≤ 32",
                })?;
                Ok(vec![Range::new(0, (1 << bits) - 1)])
            }
            Op::Argmax => {
                self.within(
                    node,
                    0,
                    -(1 << 28) + 1,
                    (1 << 28) - 1,
                    "Argmax logits have |ℓ| < 2^28 (RFC-0009 Lemma 9)",
                )?;
                self.within(node, 1, 0, 1, "a one-hot entry is 0 or 1")?;
                self.one_hot(node, 1)?;
                Ok(Vec::new())
            }
            Op::InverseCdf { .. } => {
                self.integer(node, 0)?;
                self.within(node, 1, 0, 1, "a one-hot entry is 0 or 1")?;
                self.one_hot(node, 1)?;
                Ok(Vec::new())
            }
            Op::BitDecompose { low, high } => {
                // A committed graph input may hold any value: its bits range-check it (RFC-0066
                // Lemma 1's converse), read directly or moved unchanged. Its declared range, which
                // its other readers trust, must then hold all of [0, 2^{low + high}). Any other
                // input is a true integer there.
                let top = (1i128 << (u32::from(*low) + u32::from(*high))) - 1;
                if let Some(source) = self.committed_source(node.inputs[0]) {
                    let declared = self.range(source).bounded();
                    if !declared.is_some_and(|r| r.lo <= 0 && top <= r.hi) {
                        return Err(CompileError::Range {
                            node: node.id,
                            input: 0,
                            rule: "a committed integer's declared range holds [0, 2^{low + high})",
                        });
                    }
                } else {
                    self.within(node, 0, 0, top, "the integer lies in [0, 2^{low + high})")?;
                }
                let outputs = if *high == 0 { 1 } else { 2 };
                Ok(vec![Range::new(0, 1); outputs])
            }
            Op::BitFn { f } => {
                for k in 0..f.arity() {
                    self.within(node, k, 0, 1, "a Boolean function's input is a bit")?;
                    // A committed input's declared [0, 1] proves no bit; a BitDecompose's does.
                    if self.committed_source(node.inputs[k]).is_some() {
                        return Err(CompileError::Range {
                            node: node.id,
                            input: k,
                            rule: "a Boolean function's input is a bit some operation proves",
                        });
                    }
                }
                Ok(vec![Range::new(0, 1)])
            }
            Op::AssertEqual => {
                for k in 0..2 {
                    self.integer(node, k)?;
                }
                Ok(Vec::new())
            }
        }
    }

    /// Input `k` of `node` holds rows of exactly one 1: a `OneHot`'s output, read directly or by a
    /// `Slice` or `GatherPublic` that keeps each vocabulary row whole, as the next token's rows
    /// are taken (RFC-0009 'OneHot', teacher forcing). Lemma 9's argument needs Lemma 8's "exactly
    /// one 1", which bits from any other operation, or a committed input's declared `[0, 1]`, do
    /// not give (#540).
    fn one_hot(&self, node: &Node, k: usize) -> Result<(), CompileError> {
        let refused = CompileError::Range {
            node: node.id,
            input: k,
            rule: "a one-hot input is a OneHot's output, its rows whole",
        };
        let mut tensor = node.inputs[k];
        while let Some(producer) = self.producer(tensor) {
            let input = producer.inputs[0];
            let rank = self.shape(input).len();
            match producer.op {
                Op::OneHot { .. } => return Ok(()),
                // Positions alone: not the vocabulary axis, the last.
                Op::Slice { axis, .. } if usize::from(axis) + 1 < rank => tensor = input,
                Op::GatherPublic { .. } if rank >= 2 => tensor = input,
                _ => return Err(refused),
            }
        }
        Err(refused)
    }

    /// `SoftmaxRow`'s outputs: the exponentials in its `Exp` table's values, and `R_q`, which its
    /// range check bounds by `2^{σ+1}` (RFC-0009). The scores are true integers with `|x| ≤ 2^21`,
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
            -SCORE_BOUND,
            SCORE_BOUND,
            "a score lies in [−2^21, 2^21]",
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

/// `w`, the width of `RsqrtRow`'s range check `ρ < 2^w` (RFC-0022): the bit length of the
/// largest root any row can have, `⌊√(d · 2^{2κ} / e)⌋` at a sum of 0, with a minimum of one bit
/// (RFC-0050). `None` when `e = 0` leaves
/// the roots unbounded, or when `w` exceeds the 32 bits of one lookup.
pub(crate) fn root_bits(kappa: u8, eps_int: u64, dim: u32) -> Option<u32> {
    // d < 2^32 and κ ≤ 31 keep N below 2^94.
    let big = u128::from(dim) << (2 * u32::from(kappa));
    let quotient = big.checked_div(u128::from(eps_int))?;
    // `2^k ≤ x < 2^{k+1}` puts `⌊√x⌋` in `[2^{⌊k/2⌋}, 2^{⌊k/2⌋+1})`: `⌊k/2⌋ + 1` bits.
    let bits = quotient.checked_ilog2().map_or(1, |k| k / 2 + 1);
    (bits <= 32).then_some(bits)
}

/// The number of classes a table of `kind` keys by: `r_by_class`'s for `Requant` and `Silu`, one
/// for the others.
fn classes(kind: &TableKind) -> i128 {
    match kind {
        TableKind::Requant { r_by_class, .. } | TableKind::Silu { r_by_class, .. } => {
            i128::try_from(r_by_class.len()).unwrap_or(i128::MAX)
        }
        _ => 1,
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

/// The committed graph input whose values `tensor` holds, moved but unchanged by `Reshape`,
/// `Slice`, `Transpose`, `GatherPublic` or `Expand`, if there is one, `producer` giving each
/// tensor's producing node: the walk RFC-0006 'Committed graph inputs' takes from a
/// `BitDecompose`'s input, which range analysis and the schedule's padding rule share (RFC-0072
/// 'Padded positions').
pub(crate) fn committed_source<'g>(
    graph: &'g Graph,
    producer: impl Fn(TensorId) -> Option<&'g Node>,
    mut tensor: TensorId,
) -> Option<TensorId> {
    while let Some(node) = producer(tensor) {
        match node.op {
            Op::Reshape
            | Op::Slice { .. }
            | Op::Transpose { .. }
            | Op::GatherPublic { .. }
            | Op::Expand { .. } => tensor = *node.inputs.first()?,
            _ => return None,
        }
    }
    let kind = graph.tensors.get(index(tensor)).map(|t| t.kind);
    matches!(kind, Some(NodeKind::Committed { .. })).then_some(tensor)
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
