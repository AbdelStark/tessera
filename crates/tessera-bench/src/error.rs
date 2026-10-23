// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate: an input the model or a measurement cannot use, or an output
//! that is not a number. Neither is ever coerced (04 §1).

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
    /// A name outside its alphabet or length, or one already used.
    Malformed,
    /// Too small for any plan: memory in which no group of a leaf stage fits (RFC-0054).
    TooSmall,
}

/// `is not finite`, `is negative`, `is not positive`, `is above 1`, `is empty`, `is malformed`
/// or `is too small`.
impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotFinite => "is not finite",
            Self::Negative => "is negative",
            Self::NotPositive => "is not positive",
            Self::AboveOne => "is above 1",
            Self::Empty => "is empty",
            Self::Malformed => "is malformed",
            Self::TooSmall => "is too small",
        })
    }
}

/// Why the model refused to project or a measurement to report, or could not.
///
/// The crate handles public planning and timing data only (counts, unit costs, policy
/// parameters, run times), so its errors may name the input at fault; they still carry no value.
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
