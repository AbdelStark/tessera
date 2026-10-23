// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera bench model`: the cost-model projection of RFC-0020 'Executable cost model'.
//!
//! It projects the proof multiplier `μ` term by term at each context, with the proof time of a
//! request, the sampling rate for the target amortized overhead and the detection probability
//! of the violating requests, as the source's §8 does. With `--json` it prints the
//! `tessera.cost-model/1` report on stdout, and otherwise a table; `--out` writes the report to a
//! file. `--source-defaults` takes the source's counts and unit costs, the only ones until the
//! compiler's commitment report supplies counts from a plan (`--plan`, #87).

use std::io::{self, Write as _};
use std::path::PathBuf;

use clap::{ArgGroup, Args};
use tessera_bench::{
    Counts, MemoryModel, ModelError, ModelReport, SOURCE_CONTEXTS, Scenario, UnitCosts,
};
use tessera_codec::error::{ErrorCode, TesseraError};

use crate::{io_error, write_atomically};

/// `tessera bench model`.
#[derive(Debug, Args)]
#[command(group(ArgGroup::new("counts").required(true).args(["plan", "source_defaults"])))]
pub(crate) struct ModelArgs {
    /// Counts from a compiled execution plan (#87).
    #[arg(long, value_name = "plan.tep")]
    plan: Option<PathBuf>,
    /// Use the source's counts and unit costs (source §5).
    #[arg(long)]
    source_defaults: bool,
    /// Context lengths in tokens, comma-separated [default: 1024 to 131072 by doubling].
    #[arg(long, value_delimiter = ',', value_name = "TOKENS")]
    context: Vec<u32>,
    /// GPUs that prove a request.
    #[arg(long, value_name = "G", default_value_t = Scenario::SOURCE.gpus)]
    gpus: u32,
    /// Tokens of a proof batch, which share its weight pass.
    #[arg(long, value_name = "T", default_value_t = Scenario::SOURCE.batch_tokens)]
    batch_tokens: u32,
    /// Ratio of ideal prefill time to serving time per token.
    #[arg(long, value_name = "η", default_value_t = Scenario::SOURCE.eta)]
    eta: f64,
    /// Amortized overhead the sampling rate is set for, as a fraction.
    #[arg(long, value_name = "O", default_value_t = Scenario::SOURCE.target_overhead)]
    target_overhead: f64,
    /// Violating requests whose detection probability is reported.
    #[arg(long, value_name = "V", default_value_t = Scenario::SOURCE.violations)]
    violations: u32,
    /// Write the tessera.cost-model/1 report to this file.
    #[arg(long, value_name = "MU.json")]
    out: Option<PathBuf>,
}

/// `ERR-USAGE` for an input the model refuses, `ERR-INTERNAL` for an output it could not compute.
fn model_error(error: ModelError) -> TesseraError {
    let code = match error {
        ModelError::Input { .. } => ErrorCode::Usage,
        ModelError::Output { .. } => ErrorCode::Internal,
    };
    TesseraError::new(code).with_source(error)
}

/// Runs `tessera bench model`; with `json`, the report is the result printed on stdout.
pub(crate) fn model(args: &ModelArgs, json: bool) -> Result<(), TesseraError> {
    let (source_defaults, counts) = match &args.plan {
        Some(plan) => (false, crate::compile::counts_from_plan(plan)?),
        None => (true, Counts::SOURCE_405B),
    };
    let contexts = match args.context.as_slice() {
        [] => SOURCE_CONTEXTS.to_vec(),
        given => given.to_vec(),
    };
    let scenario = Scenario {
        gpus: args.gpus,
        batch_tokens: args.batch_tokens,
        eta: args.eta,
        target_overhead: args.target_overhead,
        violations: args.violations,
    };
    let report = ModelReport::new(
        source_defaults,
        counts,
        UnitCosts::SOURCE,
        MemoryModel::NODE_405B,
        scenario,
        &contexts,
    )
    .map_err(model_error)?;
    let text = report.to_json().map_err(model_error)?;
    if let Some(path) = &args.out {
        crate::manifest::beside(path);
        write_atomically(path, &format!("{text}\n"))?;
        crate::manifest::output_report(path, "tessera.cost-model/1");
    }
    let shown = match json {
        true => format!("{text}\n"),
        false => table(&report),
    };
    io::stdout()
        .lock()
        .write_all(shown.as_bytes())
        .map_err(io_error("cannot write to stdout".to_owned()))
}

/// The projections as a table, one row per context.
fn table(report: &ModelReport) -> String {
    let mut text = format!(
        "{:>8} {:>8} {:>10} {:>8} {:>6} {:>10} {:>8} {:>8} {:>7} {:>8} {:>7} {:>8} {:>12} {:>9} {:>10}\n",
        "context",
        "linear",
        "attention",
        "weights",
        "regen",
        "streaming",
        "answers",
        "combine",
        "tables",
        "μ",
        "groups",
        "windows",
        "proof time",
        "sampling",
        "detection"
    );
    for p in &report.projections {
        text.push_str(&format!(
            "{:>8} {:>8.3} {:>10.3} {:>8.3} {:>6.3} {:>10.3} {:>8.3} {:>8.3} {:>7.3} {:>8.3} {:>7} {:>8} {:>10.2} s {:>8.3}% {:>9.3}%\n",
            p.context,
            p.mu.linear,
            p.mu.attention,
            p.mu.weights,
            p.mu.regen,
            p.mu.streaming,
            p.mu.column_answers,
            p.mu.row_combination,
            p.mu.tables,
            p.mu.total,
            p.leaf_groups,
            // The reduction stages windowed (RFC-0055), each one more regeneration.
            p.reduction_stages
                .iter()
                .filter(|w| w.window_rounds > 0)
                .count(),
            p.proof_time_s,
            100.0 * p.sampling_rate,
            100.0 * p.detection,
        ));
    }
    text
}
