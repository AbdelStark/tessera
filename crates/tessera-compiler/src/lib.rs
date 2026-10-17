// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Range analysis, rescale placement, lowering, tiling, claim-list emission and the commitment
//! report (RFC-0006, RFC-0007).
//!
//! - [`lower`] builds the graph of the integer model `D` over one sequence from the public part
//!   of the model alone (RFC-0006 'Llama lowering'), and [`validate`] checks the rest of a model,
//!   its weights and hidden channel classes, against it.
//! - [`analyze`] bounds every tensor of a graph from public data alone and checks that every
//!   gadget input that must be an integer is a true integer its gadget accepts (RFC-0006 'Range
//!   analysis'); [`check_profile`] checks the profile's constraints S1 to S3 (RFC-0005
//!   'Profiles').
//!
//! - [`select`] gives every gadget-backed node its gadget, `Requant1` or `RequantSplit` for a
//!   requantization by its key's width, and sizes and shares the tables (RFC-0006 'Gadget
//!   selection').
//!
//! Tiling and emission follow.

#![forbid(unsafe_code)]

mod error;
mod lower;
mod range;
mod select;

pub use error::CompileError;
pub use lower::{Lowering, PublicModel, check_profile, lower, validate};
pub use range::{HALF_FIELD, Range, Ranges, analyze};
pub use select::{Gadget, Selection, select};
