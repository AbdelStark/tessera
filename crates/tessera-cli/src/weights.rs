// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera commit-weights`: commits `D`'s weights under an execution plan (`02 §2.2`,
//! `RFC-0012` 'Layout', `RFC-0040`, `RFC-0041` 'The plan', `RFC-0048`).
//!
//! Writes the secret state `WEIGHTS.twc` (`0x0302`) with mode `0600` and the public commitment
//! `WEIGHTS.twr` (`0x0301`) beside it by atomic rename, and prints `R_W` (`--json` prints one
//! JSON object with `weights_root`, `epoch`, `out`, `public` and the ten matrix `roots`).

use std::fs;
use std::io::{self, Write as _};
use std::num::NonZeroU32;
use std::path::PathBuf;

use clap::Args;
use serde_json::json;
use tessera_codec::error::{ErrorCode, TesseraError};
use tessera_codec::{Public, schema};
use tessera_compiler::{ExecutionPlan, PublicModel, check_plan, validate};
use tessera_ir::IntegerModel;
use tessera_pcs::{WeightCommitment, WeightCommitmentState, WeightLayout, WeightMatrix};
use tessera_protocol::ParamSet;

use crate::{Input, io_error};

/// `tessera commit-weights`.
#[derive(Debug, Args)]
pub(crate) struct CommitWeightsArgs {
    #[arg(long, value_name = "MODEL.d")]
    pub(crate) model: PathBuf,
    #[arg(long, value_name = "plan.tep")]
    pub(crate) plan: PathBuf,
    #[arg(long, value_name = "WEIGHTS.twc")]
    pub(crate) out: PathBuf,
}

fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Runs `tessera commit-weights`; writes `args.out` (`.twc`, mode `0600`) and its sibling `.twr`,
/// and prints `R_W` (or a JSON summary when `json` is set) to stdout.
pub(crate) fn commit_weights(args: &CommitWeightsArgs, json: bool) -> Result<(), TesseraError> {
    // Beside the secret state: the operator's own (05 §6).
    crate::manifest::beside(&args.out);
    let public_out = args.out.with_extension("twr");
    if public_out == args.out {
        return Err(TesseraError::new(ErrorCode::Usage)
            .with_source("--out names the secret state (.twc), not the public commitment (.twr)"));
    }

    let model: IntegerModel = Input::new(&args.model, &schema::INTEGER_MODEL).open()?;
    let plan: ExecutionPlan = Input::new(&args.plan, &schema::EXECUTION_PLAN).open()?;

    // RFC-0048 'The assumption': discharge the public and hidden domain checks on D before
    // binding R_W.
    validate(&model).map_err(|error| TesseraError::new(error.code()).with_source(error))?;

    // RFC-0007 steps 1 and 6, RFC-0041 'The plan': the plan's parameter set and commitment
    // layout must be a defined parameter set's, and the plan's claim list and tiles must be D's.
    let Some(params) = ParamSet::defined(&plan.claims.params) else {
        return Err(TesseraError::new(ErrorCode::Usage).with_source(format!(
            "{} is not a parameter set",
            plan.claims.params.0.as_str()
        )));
    };
    crate::manifest::params(params.id, None);
    if plan.claims.layout != params.layout {
        return Err(TesseraError::new(ErrorCode::Usage)
            .with_source("the plan's commitment layout is not its parameter set's"));
    }
    tessera_codec::event!(
        INFO,
        "cli.weights_checking",
        plan = Public(args.plan.display()),
        model = Public(args.model.display());
        "tessera commit-weights: checking {plan} against {model}"
    );
    check_plan(&plan, PublicModel::of(&model), &plan.claims)
        .map_err(|error| TesseraError::new(error.code()).with_source(error))?;

    let shapes = model
        .weight_shapes()
        .map_err(|error| TesseraError::new(ErrorCode::QuantArch).with_source(error))?;
    let weights = model
        .model_weights()
        .map_err(|error| TesseraError::new(ErrorCode::QuantArch).with_source(error))?;
    let layout = WeightLayout::new(params.layout.weights, shapes)
        .map_err(|error| TesseraError::new(ErrorCode::Internal).with_source(error))?;
    let row_block = NonZeroU32::new(params.row_block()).ok_or_else(|| {
        TesseraError::new(ErrorCode::Internal).with_source("parameter set row_block is 0")
    })?;
    let epoch = 0u32;
    tessera_codec::event!(
        INFO,
        "cli.weights_committing",
        matrices = WeightMatrix::ALL.len(),
        epoch = epoch;
        "tessera commit-weights: committing {matrices} weight matrices at epoch {epoch}"
    );
    let commitment = WeightCommitment::commit(&layout, row_block, epoch, &weights)
        .map_err(|error| TesseraError::new(ErrorCode::Internal).with_source(error))?;

    let public = *commitment.public();
    let state = WeightCommitmentState::new(public).map_err(io_error(
        "cannot read the operating system's random source".to_owned(),
    ))?;
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(io_error(format!("cannot write {}", args.out.display())))?;
    }
    state
        .write(&args.out)
        .map_err(io_error(format!("cannot write {}", args.out.display())))?;
    crate::manifest::output_file(&args.out, &schema::WEIGHT_COMMITMENT_STATE);
    public
        .write(&public_out)
        .map_err(io_error(format!("cannot write {}", public_out.display())))?;
    crate::manifest::output_file(&public_out, &schema::WEIGHT_COMMITMENT_PUBLIC);
    tessera_codec::event!(
        INFO,
        "cli.weights_written",
        state = Public(args.out.display()),
        public = Public(public_out.display());
        "tessera commit-weights: wrote {state} and {public}"
    );

    let weights_root = hex(&public.weights_root());
    let shown = match json {
        true => {
            let roots: Vec<String> = public.roots.iter().map(hex).collect();
            let object = json!({
                "command": "commit-weights",
                "weights_root": weights_root,
                "epoch": public.epoch,
                "out": args.out.display().to_string(),
                "public": public_out.display().to_string(),
                "roots": roots,
            });
            format!("{object}\n")
        }
        false => format!("{weights_root}\n"),
    };
    io::stdout()
        .lock()
        .write_all(shown.as_bytes())
        .map_err(io_error("cannot write to stdout".to_owned()))?;
    Ok(())
}
