// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate. Every input-dependent condition is one of its variants, never
//! a panic (04 §1).

use core::fmt;

use tessera_sumcheck::SumcheckError;

/// A column of a LogUp instance whose length its shape fixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Column {
    /// The table's values: one per row.
    TableValues,
    /// The lookups' values: one per lookup.
    LookupValues,
    /// The multiplicities: one per leaf of the table tree, `2^{h_t}`.
    Multiplicities,
}

/// `table values`, `lookup values` or `multiplicities`.
impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TableValues => "table values",
            Self::LookupValues => "lookup values",
            Self::Multiplicities => "multiplicities",
        })
    }
}

/// Why a LogUp leaf stage or a multiplicity count was refused: before the transcript moved, and
/// before anything was computed.
///
/// [`prove_logup`](crate::prove_logup) checks its inputs in this order and returns the first
/// failure. Instance by instance: that its table has a row, that the table's values are one per
/// row, that it has a lookup, that the lookups have values exactly when the table does, one per
/// lookup, and that the multiplicities are one per leaf of the table tree. Then forest by forest,
/// the shapes [`check_forests`](tessera_sumcheck::check_forests) checks. Then the schedule: one
/// entry per tree, each naming a tree of the stage that no earlier entry names.
///
/// [`multiplicities`](crate::multiplicities) refuses a lookup of no row of its table.
///
/// The variants carry indices, lengths and column names, never a field element: an error cannot
/// carry a lookup, a value or a multiplicity, all secret-class values (05 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogUpError {
    /// An instance's table has no row.
    NoRows {
        /// The instance, counted from 0 in the order given.
        instance: usize,
    },
    /// An instance has no lookup: RFC-0008's instances have at least one.
    NoLookups {
        /// The instance, counted from 0 in the order given.
        instance: usize,
    },
    /// An instance's lookups have values and its table none, or the other way round.
    ValuesMismatch {
        /// The instance, counted from 0 in the order given.
        instance: usize,
    },
    /// A column of an instance has another length than its shape fixes.
    ColumnLength {
        /// The instance, counted from 0 in the order given.
        instance: usize,
        /// The column.
        column: Column,
        /// Its length.
        len: usize,
        /// The length its shape fixes.
        expected: usize,
    },
    /// A lookup's row is not a row of its table: the key outside the table that RFC-0008 'Failure
    /// modes' calls a range analysis bug (INV-IR-2), `ERR-INTERNAL`.
    LookupRow {
        /// The lookup, counted from 0 in leaf order.
        lookup: usize,
        /// The table's number of rows.
        rows: usize,
    },
    /// The schedule has another number of entries than the stage has trees: two per instance and
    /// one per forest.
    ScheduleLength {
        /// The entries given.
        len: usize,
        /// The stage's trees.
        expected: usize,
    },
    /// A schedule entry names no tree of the stage, or a tree an earlier entry names.
    ScheduleTree {
        /// The entry, counted from 0.
        position: usize,
    },
    /// A forest's shape, from the sumcheck engine, `forest` its position among the forests.
    Sumcheck(SumcheckError),
}

impl fmt::Display for LogUpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRows { instance } => write!(f, "the table of instance {instance} has no row"),
            Self::NoLookups { instance } => write!(f, "instance {instance} has no lookup"),
            Self::ValuesMismatch { instance } => write!(
                f,
                "instance {instance} has values on one side only, its table or its lookups"
            ),
            Self::ColumnLength {
                instance,
                column,
                len,
                expected,
            } => write!(
                f,
                "the {column} of instance {instance} are {len} where its shape gives {expected}"
            ),
            Self::LookupRow { lookup, rows } => write!(
                f,
                "lookup {lookup} names no row of its table of {rows} rows"
            ),
            Self::ScheduleLength { len, expected } => write!(
                f,
                "the schedule lists {len} trees where the stage has {expected}"
            ),
            Self::ScheduleTree { position } => write!(
                f,
                "schedule entry {position} names no tree of the stage, or one already named"
            ),
            Self::Sumcheck(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for LogUpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sumcheck(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SumcheckError> for LogUpError {
    fn from(error: SumcheckError) -> Self {
        Self::Sumcheck(error)
    }
}
