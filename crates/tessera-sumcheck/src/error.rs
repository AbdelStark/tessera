// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate. Every input-dependent condition is one of its variants, never
//! a panic (04 §1).

use core::fmt;

use crate::MAX_DEGREE;

/// Why a batch was refused before the transcript moved.
///
/// [`prove_batch`](crate::prove_batch) checks its inputs in this order and returns the first
/// failure: the degree; that there is an instance; then instance by instance, that it has a
/// table and no more tables than the degree, and table by table, that the length is a power of
/// two and equals the instance's first table's.
///
/// The variants carry indices, lengths and degrees, never a field element, so an error cannot
/// carry a secret table value or an evaluation (05 §3).
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
        }
    }
}

impl std::error::Error for SumcheckError {}
