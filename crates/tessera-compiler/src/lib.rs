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
//! - [`select`] gives every gadget-backed node its gadget, `Requant1`, `RequantSplit` or
//!   `RequantWide` for a requantization by its key's width and shift, and sizes and shares the
//!   tables (RFC-0006 'Gadget selection', RFC-0024).
//! - [`claim_list`] emits the claim list from public data alone (RFC-0007, RFC-0026), and
//!   [`report`] the commitment report (RFC-0006 'Outputs').
//! - [`derive_schedule`] derives the stage schedule from a lowering: the claims each stage leaves,
//!   their splits, the relation instances that prove them and the claim pool (RFC-0010 'Stages',
//!   RFC-0037).
//! - [`plan`] builds the execution plan from the public part of `D` and its claim list: the claim
//!   list and the tiles of [`tiles`], which partition a sequence's committed values in production
//!   order. [`ExecutionPlan::expand`] gives the tiles of one sequence with the elements each holds,
//!   and [`check_plan`] refuses a plan that is not `D`'s (RFC-0041).

#![forbid(unsafe_code)]

mod claim;
mod error;
mod lower;
mod plan;
mod range;
mod report;
pub mod schedule;
mod select;

pub use claim::claim_list;
pub use error::CompileError;
pub use lower::{Lowering, PublicModel, check_profile, lower, validate};
pub use plan::{ExecutionPlan, QueryBlock, SequenceTile, Share, Tile, check_plan, plan, tiles};
pub use range::{HALF_FIELD, Range, Ranges, analyze};
pub use report::{CommitmentReport, SCHEMA as REPORT_SCHEMA, TableUse, Tally, report, stage_count};
pub use schedule::{StageSchedule, derive_schedule};
pub use select::{Gadget, Selection, select};
