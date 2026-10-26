// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate. Every input-dependent condition is one of its variants, never
//! a panic (04 §1).

use core::fmt;

use crate::{MAX_DEGREE, Operand};

/// Why a batch, a batch of forests or a relation was refused: before the transcript moved, and
/// before anything was computed.
///
/// [`prove_batch`](crate::prove_batch) checks its inputs in this order and returns the first
/// failure: the degree; that there is an instance; then instance by instance, that it has a
/// table; for a product, no more tables than the degree; for a sum of products, at least one
/// term, and term by term between 1 and the degree factors, each naming one of its tables; then
/// table by table, that the length is a power of two and equals the instance's first table's.
///
/// [`prove_forests`](crate::prove_forests) checks its forests in this order and returns the first
/// failure: that there is a forest; then forest by forest, that its leaf count is a power of two
/// and that its height is between 1 and `log2` of its leaf count. [`prove_trees`](crate::prove_trees)
/// checks its forests the same way, by their positions among its trees.
///
/// [`FractionTree::new`](crate::FractionTree::new) checks that `2^height` leaves can be
/// allocated, then that base numerators are `2^height`.
///
/// The matmul relations check their inputs at each step and return the first failure:
/// [`MatMulWeight::new`](crate::MatMulWeight::new) and [`MatMulAct::new`](crate::MatMulAct::new),
/// operand by operand, that its shape's entries can be counted and that it has them; `tables`,
/// that the input claim's point has its coordinates; `claims`, that the instance's point has its
/// coordinates and that there are two evaluations.
///
/// [`Elementwise::new`](crate::Elementwise::new) checks, in this order, that its shape holds;
/// input by input, that its length fits the relation's variables and that a partner's bit is one
/// of them; term by term, that every factor names an input and that the term's degree is between
/// 1 and [`MAX_DEGREE`]; and that there is a term. Its `tables` and `claims` check points and
/// evaluations as the matmuls' do. [`Linear::new`](crate::Linear::new) checks that each input
/// sits within its target's variables, and [`Linear::terms`](crate::Linear::terms) that the
/// target's point has its coordinates.
///
/// [`Recomposition::new`](crate::Recomposition::new) checks its bit widths, that its instance's
/// entries can be counted, then each input's length; its `tables` and `claims` check points and
/// evaluations as the others do. [`Selection::new`](crate::Selection::new) checks that its
/// entries can be counted and `x`'s length, its `tables` the point, then member by member that
/// its rows fit the kernel's axes, and its `claim` as `Rotate`'s does.
///
/// The variants carry indices, lengths, heights, degrees and operand names, never a field
/// element, so an error cannot carry a secret table value, a leaf or an evaluation (05 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckError {
    /// The degree is outside `1..=MAX_DEGREE`: a batch's, or that of a term of an elementwise
    /// relation, its factors plus `eq`'s, 0 for a relation with no term.
    Degree {
        /// The degree asked for, or the term's.
        degree: usize,
    },
    /// The batch has no instance.
    NoInstances,
    /// An instance has no table: a product of no table has no variables to sum over.
    NoTables {
        /// The instance, counted from 0 in the order given.
        instance: usize,
    },
    /// A product instance has more tables than the degree: its product would have a higher degree
    /// than the round polynomials can carry. A sum of products bounds its terms instead
    /// ([`TermFactors`](Self::TermFactors)).
    TooManyTables {
        /// The instance, counted from 0 in the order given.
        instance: usize,
        /// Its number of tables.
        tables: usize,
        /// The batch's degree.
        degree: usize,
    },
    /// A table's length is not a power of two (zero included).
    TableLength {
        /// The instance, counted from 0 in the order given.
        instance: usize,
        /// The table, counted from 0 within its instance.
        table: usize,
        /// The length given.
        len: usize,
    },
    /// Two tables of one instance differ in length.
    TableSizes {
        /// The instance, counted from 0 in the order given.
        instance: usize,
        /// The table, counted from 0 within its instance, whose length differs from the first's.
        table: usize,
        /// Its length.
        len: usize,
        /// The length of the instance's first table.
        expected: usize,
    },
    /// A sum-of-products instance has no term.
    NoTerms {
        /// The instance, counted from 0 in the order given.
        instance: usize,
    },
    /// A term multiplies no table, or more tables than the degree: its product would have no
    /// variable, or a higher degree than the round polynomials can carry.
    TermFactors {
        /// The instance, counted from 0 in the order given.
        instance: usize,
        /// The term, counted from 0 within its instance.
        term: usize,
        /// Its number of factors.
        factors: usize,
        /// The batch's degree.
        degree: usize,
    },
    /// A term's factor names no table of its instance.
    TermFactor {
        /// The instance, counted from 0 in the order given.
        instance: usize,
        /// The term, counted from 0 within its instance.
        term: usize,
        /// The factor's position, the first out of range.
        factor: usize,
        /// The instance's number of tables.
        tables: usize,
    },
    /// The batch of forests has no forest.
    NoForests,
    /// A forest's leaf count is not a power of two (zero included): its trees and their leaves
    /// stack into `2^{s+h}` entries.
    LeafCount {
        /// The forest, counted from 0 in the order given.
        forest: usize,
        /// The leaf count given.
        len: usize,
    },
    /// A forest's height is 0, or more than `log2` of its leaf count: a tree has at least one
    /// level below its root, and a forest at least one tree.
    Height {
        /// The forest, counted from 0 in the order given.
        forest: usize,
        /// The height given.
        height: usize,
        /// `log2` of its leaf count, the largest height it admits.
        vars: usize,
    },
    /// A fraction tree's `2^height` leaves cannot be counted in a `usize`, or their entries
    /// exceed the largest allocation.
    FractionHeight {
        /// The height asked for.
        height: usize,
    },
    /// A fraction tree's numerators in the prime subfield are not one per leaf.
    FractionNumerators {
        /// The numerators given.
        len: usize,
        /// The tree's `2^height` leaves.
        expected: usize,
    },
    /// A matmul operand's shape counts `2^vars` entries, `vars` at least the bits of a `usize`:
    /// no slice has that many.
    OperandTooLarge {
        /// The operand.
        operand: Operand,
        /// The sum of its dimensions' logs of two, `usize::MAX` if it overflows.
        vars: usize,
    },
    /// A matmul operand's length is not the `2^vars` entries its shape gives.
    OperandLength {
        /// The operand.
        operand: Operand,
        /// The length given.
        len: usize,
        /// The length its shape gives.
        expected: usize,
    },
    /// A point given to a relation has the wrong number of coordinates: the input claim's point,
    /// or the instance's point after the sumcheck.
    PointLength {
        /// The number of coordinates given.
        len: usize,
        /// The number the shape gives.
        expected: usize,
    },
    /// The final evaluations given for a relation's instance are not as many as it sends: one
    /// per operand.
    EvaluationCount {
        /// The number given.
        len: usize,
        /// The number the instance sends.
        expected: usize,
    },
    /// An elementwise relation's or a `Rotate`'s shape cannot hold: it sums out more variables
    /// than it has, or its `2^vars` entries cannot be counted in a `usize`.
    RelationShape {
        /// The relation's variables, `n`.
        vars: usize,
        /// The low variables it sums out, `m`.
        summed_vars: usize,
    },
    /// A relation's input does not fit its `n` variables. In an elementwise relation an operand,
    /// a partner or a public table has `2^n` entries, and a broadcast operand `2^j` with
    /// `k + j ≤ n`, `k` the low variables it is constant in. `Rotate`'s operand has `2^n`, and
    /// its constants `2^{n−1}` each.
    InputLength {
        /// The input, counted from 0 in the order given.
        input: usize,
        /// Its length.
        len: usize,
        /// The relation's variables, `n`.
        vars: usize,
    },
    /// A relation pairs entries along a bit that is not one of its variables: an elementwise
    /// relation's partner input, or `Rotate`'s pair bit.
    PartnerBit {
        /// The input, counted from 0 in the order given.
        input: usize,
        /// The bit it flips, from 0 for the lowest variable.
        bit: usize,
        /// The relation's variables, `n`.
        vars: usize,
    },
    /// A term of an elementwise relation names an input the relation does not have.
    TermInput {
        /// The term, counted from 0 in the order given.
        term: usize,
        /// The factor's position, the first out of range.
        factor: usize,
        /// The relation's number of inputs.
        inputs: usize,
    },
    /// A `Linear` relation's input does not sit within its target's variables: a broadcast's
    /// low variables and own variables exceed them, or a sum's input has more variables than a
    /// `usize` counts.
    InputPlacement {
        /// The input, counted from 0 in the order given.
        input: usize,
        /// The target's variables.
        target_vars: usize,
    },
    /// A [`Recomposition`](crate::Recomposition)'s bit widths are not a `BitDecompose`'s
    /// (RFC-0066): `low` is 0, `low + high` exceeds 62, or its high bits are given exactly when
    /// `high` is 0.
    BitWidths {
        /// The result bits, `low`.
        low: u8,
        /// The bits above them, `high`.
        high: u8,
    },
    /// A [`Selection`](crate::Selection)'s kernel does not fit its axes (RFC-0072 'The selection
    /// relation'): a member selects more rows than the output's axis has, `len > 2^rows`, or
    /// names a row outside `x`'s axis of `2^vars`.
    SelectionRows {
        /// The member, counted from 0 in the order given.
        member: usize,
        /// The rows it selects.
        len: usize,
        /// The output axis's variables.
        rows: usize,
        /// The input axis's variables.
        vars: usize,
    },
    /// RFC-0054's merge was given inputs that do not fit its shapes: a claim on no polynomial,
    /// a point or group bits of the wrong size, a group variable out of order, a table of the
    /// wrong length, or a two-pass prover asked for more table rounds than its polynomials have.
    Merge {
        /// The rule broken.
        rule: &'static str,
    },
}

impl fmt::Display for SumcheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Degree { degree } => {
                write!(f, "a sumcheck has degree 1 to {MAX_DEGREE}, not {degree}")
            }
            Self::NoInstances => write!(f, "a batched sumcheck needs at least one instance"),
            Self::NoTables { instance } => {
                write!(f, "instance {instance} is a product of no table")
            }
            Self::TooManyTables {
                instance,
                tables,
                degree,
            } => write!(
                f,
                "instance {instance} is a product of {tables} tables, more than the degree {degree}"
            ),
            Self::TableLength {
                instance,
                table,
                len,
            } => write!(
                f,
                "table {table} of instance {instance} has {len} entries, not a power of two"
            ),
            Self::TableSizes {
                instance,
                table,
                len,
                expected,
            } => write!(
                f,
                "table {table} of instance {instance} has {len} entries where its first table has \
                 {expected}"
            ),
            Self::NoTerms { instance } => {
                write!(f, "instance {instance} is a sum of no term")
            }
            Self::TermFactors {
                instance,
                term,
                factors,
                degree,
            } => write!(
                f,
                "term {term} of instance {instance} multiplies {factors} tables, outside 1 to the \
                 degree {degree}"
            ),
            Self::TermFactor {
                instance,
                term,
                factor,
                tables,
            } => write!(
                f,
                "term {term} of instance {instance} names table {factor}, where the instance has \
                 {tables}"
            ),
            Self::NoForests => write!(f, "a batch of forests needs at least one forest"),
            Self::LeafCount { forest, len } => {
                write!(f, "forest {forest} has {len} leaves, not a power of two")
            }
            Self::Height {
                forest,
                height,
                vars,
            } => write!(
                f,
                "forest {forest} has height {height}, outside 1 to {vars}, the log2 of its leaf \
                 count"
            ),
            Self::FractionHeight { height } => write!(
                f,
                "a fraction tree of height {height} has more leaves than memory can hold"
            ),
            Self::FractionNumerators { len, expected } => write!(
                f,
                "a fraction tree has {len} numerators where it has {expected} leaves"
            ),
            Self::OperandTooLarge { operand, vars } => write!(
                f,
                "operand {operand} has 2^{vars} entries by its shape, more than a usize counts"
            ),
            Self::OperandLength {
                operand,
                len,
                expected,
            } => write!(
                f,
                "operand {operand} has {len} entries where its shape gives {expected}"
            ),
            Self::PointLength { len, expected } => {
                write!(
                    f,
                    "a point has {len} coordinates where the shape gives {expected}"
                )
            }
            Self::EvaluationCount { len, expected } => write!(
                f,
                "{len} final evaluations where the instance sends {expected}"
            ),
            Self::RelationShape { vars, summed_vars } => write!(
                f,
                "a relation on {vars} variables cannot sum out {summed_vars} of them, or count \
                 2^{vars} entries"
            ),
            Self::InputLength { input, len, vars } => write!(
                f,
                "input {input} has {len} entries, which do not fit a relation on {vars} variables"
            ),
            Self::PartnerBit { input, bit, vars } => write!(
                f,
                "input {input} flips bit {bit}, outside a relation on {vars} variables"
            ),
            Self::TermInput {
                term,
                factor,
                inputs,
            } => write!(
                f,
                "factor {factor} of term {term} names no input, where the relation has {inputs}"
            ),
            Self::InputPlacement { input, target_vars } => write!(
                f,
                "input {input} does not sit within a target of {target_vars} variables"
            ),
            Self::BitWidths { low, high } => write!(
                f,
                "a recomposition of {low} low and {high} high bits is not a BitDecompose's"
            ),
            Self::SelectionRows {
                member,
                len,
                rows,
                vars,
            } => write!(
                f,
                "selection {member} takes {len} rows, which do not fit an output axis of {rows} \
                 variables and an input axis of {vars}"
            ),
            Self::Merge { rule } => write!(f, "the merge's inputs break a rule: {rule}"),
        }
    }
}

impl std::error::Error for SumcheckError {}
