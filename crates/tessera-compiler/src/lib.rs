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
//! - [`ModelAdviceLayout`] binds every committed graph tensor and RFC-0050 auxiliary to the canonical PCS
//!   allocation, zero rows and segments. It maps whole attention claims to weighted share
//!   openings. [`claim_list`] checks the complete allocation at `max_context` before emission
//!   and returns an error when it cannot fit (RFC-0046/0047).

#![forbid(unsafe_code)]

mod advice;
mod claim;
mod error;
mod lower;
mod plan;
mod range;
mod report;
mod runtime;
pub mod schedule;
mod select;

pub use advice::{AdviceClaimImage, AdviceSource, AuxColumn, CompareColumn, ModelAdviceLayout};
pub use claim::claim_list;
pub use error::CompileError;
pub use lower::{Lowering, PublicModel, check_profile, lower, validate};
pub use plan::{ExecutionPlan, QueryBlock, SequenceTile, Share, Tile, check_plan, plan, tiles};
pub use range::{HALF_FIELD, Range, Ranges, analyze};
pub use report::{
    CommitmentReport, Growth, Holding, SCHEMA as REPORT_SCHEMA, StageTables, TableClass, TableUse,
    Tally, report, stage_count,
};
pub use runtime::{GreedyCertificate, RuntimePlan};
pub use schedule::{StageSchedule, derive_schedule};
pub use select::{Gadget, Selection, select};

/// The three outputs of compiling `D` (RFC-0006 'Outputs', RFC-0029): the public
/// [`tessera_ir::ClaimList`], the private [`ExecutionPlan`], and the public [`CommitmentReport`].
#[derive(Clone, Debug, PartialEq)]
pub struct CompileOutput {
    /// The public claim list (RFC-0007); its tables are the table set (RFC-0029).
    pub claim_list: tessera_ir::ClaimList,
    /// The prover-private execution plan (RFC-0041).
    pub plan: ExecutionPlan,
    /// What the lowered model commits and looks up (RFC-0006 'Outputs').
    pub report: CommitmentReport,
}

/// Compiles `model` under `policy`, for the parameter set `params` with its `layout` (RFC-0006
/// 'Outputs').
///
/// Validates `model`, lowers it, analyzes ranges, selects gadgets and tables, emits the claim
/// list, builds the execution plan, and counts the commitment report.
///
/// # Errors
///
/// Returns [`CompileError`] if `model` fails validation, range analysis, profile checks, gadget
/// selection, advice layout bounds, or plan construction.
pub fn compile(
    model: &tessera_ir::IntegerModel,
    policy: &tessera_ir::GadgetPolicy,
    layout: tessera_ir::CommitmentLayout,
    params: tessera_ir::ParamSetId,
) -> Result<CompileOutput, CompileError> {
    let _compile = tessera_codec::span!(
        INFO,
        "compile",
        params = tessera_codec::Public(params.0.as_str()),
        layers = model.arch.n_layers
    )
    .entered();
    validate(model)?;
    let public = PublicModel::of(model);
    let positions = 16.min(model.arch.max_context).max(1);
    let lowering = lower(public, positions)?;
    let ranges = analyze(&lowering.graph, &lowering.tables)?;
    let selection = select(&lowering.graph, &ranges, &lowering.tables, policy)?;
    let claims = claim_list(public, &selection, policy, layout, params)?;
    let execution_plan = plan(public, &claims)?;
    let commitment_report = report(public, &lowering, &selection, policy)?;
    tessera_codec::event!(
        DEBUG,
        "compiler.compiled",
        tables = claims.tables.len(),
        gadgets = selection.gadgets.len();
        "{tables} tables, {gadgets} gadgets"
    );
    Ok(CompileOutput {
        claim_list: claims,
        plan: execution_plan,
        report: commitment_report,
    })
}
