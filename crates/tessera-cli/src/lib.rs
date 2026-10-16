// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `tessera` command-line interface (docs/spec/02-public-api.md).
//!
//! This version has one command, `tessera bench model`, the cost-model projection of
//! RFC-0020 'Executable cost model'; the other commands of 02 §2.2 come with their components.
//!
//! ```text
//! tessera bench model --source-defaults [--context TOKENS,...] [--gpus G] [--batch-tokens T]
//!                     [--eta η] [--target-overhead O] [--violations V] [--out MU.json] [--json]
//! ```
//!
//! It projects the proof multiplier `μ` term by term at each context, with the proof time of a
//! request, the sampling rate for the target amortized overhead and the detection probability
//! of the violating requests, as the source's §8 does. With `--json` it prints the
//! `tessera.cost-model/1` report on stdout, and otherwise a table; `--out` writes the report to a
//! file. `--source-defaults` is required: it takes the source's counts and unit costs, the only
//! ones until the compiler's commitment report can supply counts (`--plan`).
//!
//! Exit codes follow 02 §2.4 through the error codes of 04 §3: 0 on success, 64 for a usage
//! error (`ERR-USAGE`), 20 when the report cannot be written (`ERR-RES-IO`), 70 for an internal
//! error (`ERR-INTERNAL`). Errors go to stderr as `error: ERR-...: what happened`.

#![forbid(unsafe_code)]
// A command line writes its results to stdout and its errors to stderr (02 §2).
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::ffi::OsString;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use tessera_bench::{
    Counts, MemoryModel, ModelError, ModelReport, SOURCE_CONTEXTS, Scenario, UnitCosts,
};
use tessera_codec::error::{ErrorCode, TesseraError};

/// The `tessera` command tree.
#[derive(Debug, Parser)]
#[command(
    name = "tessera",
    version,
    about = "The Tessera prover (docs/spec/02-public-api.md)."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Benchmarks and the cost model (RFC-0020).
    #[command(subcommand)]
    Bench(Bench),
}

#[derive(Debug, Subcommand)]
enum Bench {
    /// Projects the proof multiplier μ with the cost model of 08 §1.
    Model(ModelArgs),
}

/// `tessera bench model`.
#[derive(Debug, Args)]
struct ModelArgs {
    /// Use the source's counts and unit costs (source §5). Required until the compiler's
    /// commitment report supplies counts.
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
    /// Print the report as one JSON object on stdout instead of a table.
    #[arg(long)]
    json: bool,
}

/// Runs `tessera` on `args`, the program's name first, and returns its exit code (02 §2.4).
pub fn run<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) => {
            // Help and version go to stdout and succeed; every other parse failure is a usage
            // error, 64 rather than clap's 2.
            let _ = error.print();
            return match error.use_stderr() {
                true => ExitCode::from(ErrorCode::Usage.exit_code()),
                false => ExitCode::SUCCESS,
            };
        }
    };
    let outcome = match cli.command {
        Command::Bench(Bench::Model(args)) => bench_model(&args),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            match &error.source {
                Some(source) => eprintln!("error: {error}: {source}"),
                None => eprintln!("error: {error}"),
            }
            ExitCode::from(error.exit_code())
        }
    }
}

/// `ERR-USAGE` for an input the model refuses, `ERR-INTERNAL` for an output it could not compute.
fn model_error(error: ModelError) -> TesseraError {
    let code = match error {
        ModelError::Input { .. } => ErrorCode::Usage,
        ModelError::Output { .. } => ErrorCode::Internal,
    };
    TesseraError::new(code).with_source(error)
}

fn io_error(what: String) -> impl FnOnce(io::Error) -> TesseraError {
    move |error| TesseraError::new(ErrorCode::ResIo).with_source(format!("{what}: {error}"))
}

fn bench_model(args: &ModelArgs) -> Result<(), TesseraError> {
    if !args.source_defaults {
        return Err(TesseraError::new(ErrorCode::Usage).with_source(
            "tessera bench model needs --source-defaults: counts from a compiled plan come with \
             the compiler's commitment report"
                .to_owned(),
        ));
    }
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
        true,
        Counts::SOURCE_405B,
        UnitCosts::SOURCE,
        MemoryModel::NODE_405B,
        scenario,
        &contexts,
    )
    .map_err(model_error)?;
    let json = report.to_json().map_err(model_error)?;
    if let Some(path) = &args.out {
        write_atomically(path, &format!("{json}\n"))?;
    }
    let text = match args.json {
        true => format!("{json}\n"),
        false => table(&report),
    };
    io::stdout()
        .lock()
        .write_all(text.as_bytes())
        .map_err(io_error("cannot write to stdout".to_owned()))
}

/// Writes `contents` to `path` through a temporary file beside it, renamed on completion, so the
/// path holds either the old file or the whole new one (04 §6).
fn write_atomically(path: &Path, contents: &str) -> Result<(), TesseraError> {
    let shown = path.display().to_string();
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(".tmp-{}", std::process::id()));
    let temporary = PathBuf::from(temporary);
    fs::write(&temporary, contents).map_err(io_error(format!("cannot write {shown}")))?;
    fs::rename(&temporary, path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        io_error(format!("cannot write {shown}"))(error)
    })
}

/// The projections as a table, one row per context.
fn table(report: &ModelReport) -> String {
    let mut text = format!(
        "{:>8} {:>8} {:>10} {:>8} {:>6} {:>10} {:>8} {:>7} {:>12} {:>9} {:>10}\n",
        "context",
        "linear",
        "attention",
        "weights",
        "regen",
        "streaming",
        "μ",
        "passes",
        "proof time",
        "sampling",
        "detection"
    );
    for p in &report.projections {
        text.push_str(&format!(
            "{:>8} {:>8.3} {:>10.3} {:>8.3} {:>6.3} {:>10.3} {:>8.3} {:>7} {:>10.2} s {:>8.3}% {:>9.3}%\n",
            p.context,
            p.mu.linear,
            p.mu.attention,
            p.mu.weights,
            p.mu.regen,
            p.mu.streaming,
            p.mu.total,
            p.streaming_passes,
            p.proof_time_s,
            100.0 * p.sampling_rate,
            100.0 * p.detection,
        ));
    }
    text
}
