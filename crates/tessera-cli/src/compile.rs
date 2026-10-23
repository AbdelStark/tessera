// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera compile`: compiles an `IntegerModel` (`D`) into `claims.tcl`, `plan.tep` and
//! `commitment-report.json` (02 §2.2, RFC-0006 'Outputs', RFC-0029).

use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use clap::Args;
use serde_json::Value;
use tessera_bench::{Counts, MemoryModel, ModelReport, Scenario, UnitCosts};
use tessera_codec::error::{ErrorCode, TesseraError};
use tessera_codec::{Ascii, Public, envelope, schema};
use tessera_compiler::{
    CommitmentReport, CompileError, ExecutionPlan, PublicModel, analyze, check_plan, lower, report,
    select,
};
use tessera_ir::{GadgetPolicy, IntegerModel, ParamSetId};
use tessera_protocol::ParamSet;

use crate::{Input, PARAM_SETS, io_error, write_atomically, write_bytes_atomically};

/// `tessera compile`.
#[derive(Debug, Args)]
pub(crate) struct CompileArgs {
    #[arg(long, value_name = "MODEL.d")]
    pub(crate) model: PathBuf,
    #[arg(long, value_name = "PARAMSET", value_parser = PARAM_SETS)]
    pub(crate) params: String,
    /// Where claims.tcl, plan.tep and commitment-report.json go.
    #[arg(long, value_name = "DIR")]
    pub(crate) out_dir: PathBuf,
    /// Largest lookup table, as log2(rows); wider keys use the existing split gadgets.
    #[arg(long,value_parser=clap::value_parser!(u8).range(1..=32),default_value_t=32)]
    pub(crate) max_log2_rows: u8,
    /// Positions per attention tile; use 2 for the resident tiny-model reference path.
    #[arg(long,value_parser=clap::value_parser!(u32).range(1..),default_value_t=512)]
    pub(crate) query_block: u32,
}

fn compile_error(error: CompileError) -> TesseraError {
    TesseraError::new(error.code()).with_source(error)
}

/// Converts a [`CommitmentReport`]'s counts into `tessera-bench`'s [`Counts`].
pub(crate) fn counts_of(report: &CommitmentReport) -> Counts {
    let counts = report.counts();
    let number = |key: &str| counts[key].as_f64().unwrap_or(0.0);
    Counts {
        commits_per_token: number("commits_per_token"),
        lookups_per_token: number("lookups_per_token"),
        scores_per_token_per_ctx: number("scores_per_token_per_ctx"),
        native_ops_lin: number("native_ops_lin"),
        native_ops_att_per_ctx: number("native_ops_att_per_ctx"),
        regen_full_equivalents: 5.0,
        table_rows: number("table_rows"),
        score_block: number("score_block"),
    }
}

/// Derives the cost-model [`Counts`] of a compiled [`ExecutionPlan`] (`plan.tep`).
pub(crate) fn counts_from_plan(path: &Path) -> Result<Counts, TesseraError> {
    let plan: ExecutionPlan = Input::new(path, &schema::EXECUTION_PLAN).open()?;
    let public = PublicModel::of_claims(&plan.claims);
    check_plan(&plan, public, &plan.claims).map_err(compile_error)?;
    let positions = 16.min(plan.claims.arch.max_context).max(1);
    let lowering = lower(public, positions).map_err(compile_error)?;
    let ranges = analyze(&lowering.graph, &lowering.tables).map_err(compile_error)?;
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &plan.claims.gadget_policy,
    )
    .map_err(compile_error)?;
    let commitment_report =
        report(public, &lowering, &selection, &plan.claims.gadget_policy).map_err(compile_error)?;
    Ok(counts_of(&commitment_report))
}

/// Runs `tessera compile`; writes `claims.tcl`, `plan.tep` and `commitment-report.json` to
/// `args.out_dir`, and with `json` prints the `tessera.commitment-report/1` report on stdout.
pub(crate) fn compile(args: &CompileArgs, json: bool) -> Result<(), TesseraError> {
    let model: IntegerModel = Input::new(&args.model, &schema::INTEGER_MODEL).open()?;
    let Some(params) = ParamSet::ALL
        .into_iter()
        .find(|set| set.name() == args.params)
    else {
        return Err(TesseraError::new(ErrorCode::Usage)
            .with_source(format!("{} is not a parameter set", args.params)));
    };
    let param_id = ParamSetId(Ascii::new(params.id).map_err(|_| {
        TesseraError::new(ErrorCode::Internal).with_source("parameter set id is not ASCII")
    })?);
    let policy = GadgetPolicy {
        max_log2_rows: args.max_log2_rows,
        query_block: args.query_block,
        ..GadgetPolicy::default()
    };
    let output = tessera_compiler::compile(&model, &policy, params.layout, param_id)
        .map_err(compile_error)?;

    let cost_model = ModelReport::new(
        false,
        counts_of(&output.report),
        UnitCosts::SOURCE,
        MemoryModel::NODE_405B,
        Scenario::SOURCE,
        &[1024, 4096, 16_384, 131_072],
    )
    .map_err(|error| TesseraError::new(ErrorCode::Internal).with_source(error))?;
    let cost_model_json: Value = serde_json::from_str(
        &cost_model
            .to_json()
            .map_err(|error| TesseraError::new(ErrorCode::Internal).with_source(error))?,
    )
    .map_err(|error| TesseraError::new(ErrorCode::Internal).with_source(error))?;
    let report_value = output.report.to_json(&cost_model_json);
    let report_text = serde_json::to_string_pretty(&report_value)
        .map_err(|error| TesseraError::new(ErrorCode::Internal).with_source(error))?;

    let dir = &args.out_dir;
    fs::create_dir_all(dir).map_err(io_error(format!("cannot write {}", dir.display())))?;
    let claims_bytes = envelope::seal(&schema::CLAIM_LIST, &output.claim_list);
    let plan_bytes = envelope::seal(&schema::EXECUTION_PLAN, &output.plan);
    write_bytes_atomically(&dir.join("claims.tcl"), &claims_bytes)?;
    write_bytes_atomically(&dir.join("plan.tep"), &plan_bytes)?;
    write_atomically(
        &dir.join("commitment-report.json"),
        &format!("{report_text}\n"),
    )?;

    tessera_codec::event!(
        INFO,
        "cli.compile_written",
        dir = Public(dir.display());
        "tessera compile: wrote claims.tcl, plan.tep, commitment-report.json to {dir}"
    );
    if json {
        io::stdout()
            .lock()
            .write_all(format!("{report_text}\n").as_bytes())
            .map_err(io_error("cannot write to stdout".to_owned()))?;
    }
    Ok(())
}
