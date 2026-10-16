// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate. Every input-dependent condition is one of its variants, never
//! a panic (04 §1).

use core::fmt;

use crate::{MAX_DEGREE, Operand};

/// Why a batch, a batch of forests or a matmul relation was refused: before the transcript moved,
/// and before anything was computed.
///
/// [`prove_batch`](crate::prove_batch) checks its inputs in this order and returns the first
/// failure: the degree; that there is an instance; then instance by instance, that it has a
/// table and no more tables than the degree, and table by table, that the length is a power of
/// two and equals the instance's first table's.
///
/// [`prove_forests`](crate::prove_forests) checks its forests in this order and returns the first
/// failure: that there is a forest; then forest by forest, that its leaf count is a power of two
/// and that its height is between 1 and `log2` of its leaf count.
///
/// The matmul relations check their inputs at each step and return the first failure:
/// [`MatMulWeight::new`](crate::MatMulWeight::new) and [`MatMulAct::new`](crate::MatMulAct::new),
/// operand by operand, that its shape's entries can be counted and that it has them; `tables`,
/// that the input claim's point has its coordinates; `claims`, that the instance's point has its
/// coordinates and that there are two evaluations.
///
/// The variants carry indices, lengths, heights, degrees and operand names, never a field
/// element, so an error cannot carry a secret table value, a leaf or an evaluation (05 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckError {
    /// The degree is outside `1..=MAX_DEGREE`.
    Degree {
        /// The degree asked for.
        degree: usize,
    },
    /// The batch has no instance.
    NoInstances,
    /// An instance has no table: a product of no table has no variables to sum over.
    NoTables {
        /// The instance, counted from 0 in the order given.
        instance: usize,
    },
    /// An instance has more tables than the degree: its product would have a higher degree than
    /// the round polynomials can carry.
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
    /// A point given to a matmul relation has the wrong number of coordinates: the input claim's
    /// point, or the instance's point after the sumcheck.
    PointLength {
        /// The number of coordinates given.
        len: usize,
        /// The number the shape gives.
        expected: usize,
    },
    /// The final evaluations given for a matmul's instance are not as many as it sends: one per
    /// operand.
    EvaluationCount {
        /// The number given.
        len: usize,
        /// The number the instance sends.
        expected: usize,
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
        }
    }
}

impl std::error::Error for SumcheckError {}
