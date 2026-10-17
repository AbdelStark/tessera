// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Range analysis, rescale placement, lowering, tiling, claim-list emission and the commitment
//! report (RFC-0006, RFC-0007).
//!
//! [`analyze`] bounds every tensor of a graph from public data alone and checks that every gadget
//! input that must be an integer is a true integer its gadget accepts (RFC-0006 'Range
//! analysis'). Lowering, gadget selection and emission follow.

#![forbid(unsafe_code)]

mod error;
mod range;

pub use error::CompileError;
pub use range::{HALF_FIELD, Range, Ranges, analyze};
