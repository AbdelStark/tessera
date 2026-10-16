// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate: an input the model cannot price, or an output that is not a
//! number. The model never coerces either (04 §1).

use core::fmt;

/// What is wrong with an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Problem {
    /// Not a finite number: infinite or NaN.
    NotFinite,
    /// Below zero, where only a non-negative quantity makes sense.
    Negative,
    /// Zero or below, where the model divides by it.
    NotPositive,
    /// Above 1, where only a probability makes sense.
    AboveOne,
    /// An empty list.
    Empty,
}

/// `is not finite`, `is negative`, `is not positive`, `is above 1` or `is empty`.
impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotFinite => "is not finite",
            Self::Negative => "is negative",
            Self::NotPositive => "is not positive",
            Self::AboveOne => "is above 1",
            Self::Empty => "is empty",
        })
    }
}

/// Why the model refused to project, or could not.
///
/// The cost model handles public planning data only (counts, unit costs, policy parameters), so
/// its errors may name the input at fault; they still carry no value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelError {
    /// An input the model cannot price: a usage error.
    Input {
        /// The input, as its field is named.
        name: &'static str,
        /// What is wrong with it.
        problem: Problem,
    },
    /// An output that is not a finite number, from finite inputs: an overflow or a bug.
    Output {
        /// The output, as its field is named.
        name: &'static str,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input { name, problem } => write!(f, "the input {name} {problem}"),
            Self::Output { name } => write!(f, "the output {name} is not a finite number"),
        }
    }
}

impl std::error::Error for ModelError {}
