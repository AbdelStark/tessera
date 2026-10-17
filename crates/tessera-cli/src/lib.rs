// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `tessera` command-line interface (docs/spec/02-public-api.md §2.2).
//!
//! The command tree is 02 §2.2's, every command with its arguments:
//!
//! ```text
//! tessera compile         --model MODEL.d --params PARAMSET --out-dir DIR
//! tessera commit-weights  --model MODEL.d --plan plan.tep --out WEIGHTS.twc
//! tessera weights refresh --weights WEIGHTS.twc --out WEIGHTS2.twc
//! tessera prove           --plan plan.tep --model MODEL.d --weights WEIGHTS.twc --sample SAMPLE.tas
//!                         --transcripts DIR (--live ENDPOINT | --out PROOF.tnp)
//!                         [--backend {cpu,gpu}] [--devices 0,1,...] [--memory-budget GIB]
//! tessera audit sample    --manifest MANIFEST.tlm --seed SEED.tsd --rate FLOAT --out SAMPLE.tas
//! tessera audit batch     --sample SAMPLE.tas [--min-tokens 16384] --out-dir DIR
//! tessera audit plan      --policy POLICY.json --traffic PROFILE.json --plan plan.tep
//! tessera params check    --params PARAMSET [--lookups N] [--table-rows T]
//! tessera bench unit-costs --device INT --out UNIT_COSTS.json
//! tessera bench model     (--plan plan.tep | --source-defaults) [--context INT,...] [--out MU.json]
//! tessera recorder ref    --listen ADDR --log-dir DIR
//! ```
//!
//! `tessera bench model` runs, in the `bench` module. Every other command checks its arguments
//! and the envelope of each TCE artifact it reads (RFC-0003), then exits with `ERR-USAGE`, naming
//! the issue that implements it.
//!
//! Every command prints its progress to stderr. With `--json` it prints one JSON object to
//! stdout describing its result, or `{"command", "error": {"code", "message"}}` when it fails
//! (02 §2). The exit status is the error code's (02 §2.4, 04 §3): 0 on success, 64 for a usage
//! error, 65 for a malformed artifact, 20 when an artifact cannot be read or written, 70 for an
//! internal error. Errors go to stderr as `error: ERR-...: what happened`; messages hold public
//! data only (05 §3).
//!
//! Every command's outcome leaves through [`run`]'s one exit point, where the run manifest of
//! 05 §6 will be written (#95).

#![forbid(unsafe_code)]
// A command line writes its results to stdout and its errors to stderr (02 §2).
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod bench;

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{ArgGroup, Args, CommandFactory, FromArgMatches, Parser, Subcommand};
use tessera_codec::DecodeLimits;
use tessera_codec::envelope::{self, ReadError};
use tessera_codec::error::{ErrorCode, TesseraError};
use tessera_codec::schema::{self, Schema};

/// The parameter sets of RFC-0014 'Parameter sets' (02 §5).
const PARAM_SETS: [&str; 3] = ["live-64", "ni-100", "test-small"];

/// The `tessera` command tree.
#[derive(Debug, Parser)]
#[command(
    name = "tessera",
    version,
    about = "The Tessera prover (docs/spec/02-public-api.md)."
)]
struct Cli {
    /// Print the result, or the error, as one JSON object on stdout.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compiles D into its claim list, execution plan, tables and commitment report.
    Compile(CompileArgs),
    /// Commits D's weights under an execution plan (RFC-0012).
    CommitWeights(CommitWeightsArgs),
    /// Weight-commitment maintenance (RFC-0012).
    #[command(subcommand)]
    Weights(Weights),
    /// Proves a sample of recorded requests, live or non-interactively (RFC-0014).
    Prove(ProveArgs),
    /// Audit sampling, batching and planning (RFC-0015).
    #[command(subcommand)]
    Audit(Audit),
    /// Parameter sets (RFC-0014).
    #[command(subcommand)]
    Params(Params),
    /// Benchmarks and the cost model (RFC-0020).
    #[command(subcommand)]
    Bench(Bench),
    /// Transcript recorders (RFC-0015).
    #[command(subcommand)]
    Recorder(Recorder),
}

#[derive(Debug, Subcommand)]
enum Weights {
    /// Re-masks a weight commitment and emits the proof linking it to the old root.
    Refresh(RefreshArgs),
}

#[derive(Debug, Subcommand)]
enum Audit {
    /// Draws the audit sample of a log manifest from a coin-tossed seed.
    Sample(SampleArgs),
    /// Groups a sample's requests into proof batches.
    Batch(BatchArgs),
    /// Reports an audit policy's detection probability and overhead.
    Plan(PlanArgs),
}

#[derive(Debug, Subcommand)]
enum Params {
    /// Reports a parameter set's soundness.
    Check(CheckArgs),
}

#[derive(Debug, Subcommand)]
enum Bench {
    /// Measures the unit costs of the cost model on a GPU (RFC-0020).
    UnitCosts(UnitCostsArgs),
    /// Projects the proof multiplier μ with the cost model of 08 §1.
    Model(bench::ModelArgs),
}

#[derive(Debug, Subcommand)]
enum Recorder {
    /// Runs the software reference recorder.
    Ref(RecorderArgs),
}

/// `tessera compile`.
#[derive(Debug, Args)]
struct CompileArgs {
    #[arg(long, value_name = "MODEL.d")]
    model: PathBuf,
    #[arg(long, value_name = "PARAMSET", value_parser = PARAM_SETS)]
    params: String,
    /// Where claims.tcl, plan.tep and commitment-report.json go.
    #[arg(long, value_name = "DIR")]
    out_dir: PathBuf,
}

/// `tessera commit-weights`.
#[derive(Debug, Args)]
struct CommitWeightsArgs {
    #[arg(long, value_name = "MODEL.d")]
    model: PathBuf,
    #[arg(long, value_name = "plan.tep")]
    plan: PathBuf,
    #[arg(long, value_name = "WEIGHTS.twc")]
    out: PathBuf,
}

/// `tessera weights refresh`.
#[derive(Debug, Args)]
struct RefreshArgs {
    #[arg(long, value_name = "WEIGHTS.twc")]
    weights: PathBuf,
    #[arg(long, value_name = "WEIGHTS2.twc")]
    out: PathBuf,
}

/// `tessera prove`.
#[derive(Debug, Args)]
#[command(group(ArgGroup::new("mode").required(true).args(["live", "out"])))]
struct ProveArgs {
    #[arg(long, value_name = "plan.tep")]
    plan: PathBuf,
    #[arg(long, value_name = "MODEL.d")]
    model: PathBuf,
    #[arg(long, value_name = "WEIGHTS.twc")]
    weights: PathBuf,
    #[arg(long, value_name = "SAMPLE.tas")]
    sample: PathBuf,
    /// The recorded transcripts.
    #[arg(long, value_name = "DIR")]
    transcripts: PathBuf,
    /// Prove live against a verifier at this endpoint.
    #[arg(long, value_name = "ENDPOINT")]
    live: Option<String>,
    /// Write a non-interactive proof.
    #[arg(long, value_name = "PROOF.tnp")]
    out: Option<PathBuf>,
    #[arg(long, value_parser = ["cpu", "gpu"], default_value = "cpu")]
    backend: String,
    /// GPU ordinals, comma-separated.
    #[arg(long, value_delimiter = ',', value_name = "0,1,...")]
    devices: Vec<u32>,
    /// Device memory the prover may use, in GiB.
    #[arg(long, value_name = "GIB", value_parser = clap::value_parser!(u64).range(1..))]
    memory_budget: Option<u64>,
}

/// `tessera audit sample`.
#[derive(Debug, Args)]
struct SampleArgs {
    #[arg(long, value_name = "MANIFEST.tlm")]
    manifest: PathBuf,
    /// The coin-tossed seed.
    #[arg(long, value_name = "SEED.tsd")]
    seed: PathBuf,
    /// The sampling rate, in (0, 1].
    #[arg(long, value_name = "FLOAT", value_parser = rate)]
    rate: f64,
    #[arg(long, value_name = "SAMPLE.tas")]
    out: PathBuf,
}

/// `tessera audit batch`.
#[derive(Debug, Args)]
struct BatchArgs {
    #[arg(long, value_name = "SAMPLE.tas")]
    sample: PathBuf,
    /// The fewest tokens a batch proves.
    #[arg(long, value_name = "TOKENS", default_value_t = 16384,
          value_parser = clap::value_parser!(u64).range(1..))]
    min_tokens: u64,
    #[arg(long, value_name = "DIR")]
    out_dir: PathBuf,
}

/// `tessera audit plan`.
#[derive(Debug, Args)]
struct PlanArgs {
    #[arg(long, value_name = "POLICY.json")]
    policy: PathBuf,
    #[arg(long, value_name = "PROFILE.json")]
    traffic: PathBuf,
    #[arg(long, value_name = "plan.tep")]
    plan: PathBuf,
}

/// `tessera params check`.
#[derive(Debug, Args)]
struct CheckArgs {
    #[arg(long, value_name = "PARAMSET", value_parser = PARAM_SETS)]
    params: String,
    /// Lookups the soundness bound counts.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..))]
    lookups: Option<u64>,
    /// Rows of the largest table.
    #[arg(long, value_name = "T", value_parser = clap::value_parser!(u64).range(1..))]
    table_rows: Option<u64>,
}

/// `tessera bench unit-costs`.
#[derive(Debug, Args)]
struct UnitCostsArgs {
    /// The GPU ordinal.
    #[arg(long, value_name = "INT")]
    device: u32,
    #[arg(long, value_name = "UNIT_COSTS.json")]
    out: PathBuf,
}

/// `tessera recorder ref`.
#[derive(Debug, Args)]
struct RecorderArgs {
    #[arg(long, value_name = "ADDR")]
    listen: SocketAddr,
    #[arg(long, value_name = "DIR")]
    log_dir: PathBuf,
}

/// The command tree, its help ending with every command, as 02 §2.2 lists them.
fn command() -> clap::Command {
    fn leaves(command: &clap::Command, path: &str, out: &mut Vec<(String, String)>) {
        for sub in command.get_subcommands().filter(|c| c.get_name() != "help") {
            let here = format!("{path} {}", sub.get_name());
            match sub.has_subcommands() {
                true => leaves(sub, &here, out),
                false => {
                    let about = sub.get_about().map(ToString::to_string).unwrap_or_default();
                    out.push((here, about));
                }
            }
        }
    }
    let tree = Cli::command();
    let mut every = Vec::new();
    leaves(&tree, "tessera", &mut every);
    let width = every.iter().map(|(path, _)| path.len()).max().unwrap_or(0);
    let listing: Vec<String> = every
        .iter()
        .map(|(path, about)| format!("  {path:<width$}  {about}"))
        .collect();
    tree.after_help(format!("Every command:\n{}", listing.join("\n")))
}

/// A sampling rate: a number in `(0, 1]`.
fn rate(text: &str) -> Result<f64, String> {
    let value: f64 = text.parse().map_err(|_| format!("not a number: {text}"))?;
    match value > 0.0 && value <= 1.0 {
        true => Ok(value),
        false => Err("a rate lies in (0, 1]".to_owned()),
    }
}

/// Runs `tessera` on `args`, the program's name first, and returns its exit status (02 §2.4).
pub fn run<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let json = args.iter().skip(1).any(|a| a == "--json");
    let parsed = command()
        .try_get_matches_from(&args)
        .and_then(|matches| Cli::from_arg_matches(&matches));
    let cli = match parsed {
        Ok(cli) => cli,
        Err(error) => {
            // Help and version go to stdout and succeed; every other parse failure is a usage
            // error, 64 rather than clap's 2.
            if !error.use_stderr() {
                let _ = error.print();
                return ExitCode::SUCCESS;
            }
            // clap's first line, without its `error: ` prefix, is the message.
            let rendered = error.to_string();
            let first = rendered.lines().next().unwrap_or_default();
            let message = first.strip_prefix("error: ").unwrap_or(first).to_owned();
            let _ = error.print();
            return fail(
                None,
                json,
                &TesseraError::new(ErrorCode::Usage),
                &message,
                false,
            );
        }
    };
    let (name, outcome) = dispatch(&cli);
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let message = error
                .source
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default();
            fail(Some(name), cli.json, &error, &message, true)
        }
    }
}

/// Reports a failure: on stderr, and with `json` as an object on stdout. `shown` is whether
/// stderr still needs it; clap has printed its own.
fn fail(
    command: Option<&str>,
    json: bool,
    error: &TesseraError,
    message: &str,
    shown: bool,
) -> ExitCode {
    if shown {
        match message.is_empty() {
            true => eprintln!("error: {error}"),
            false => eprintln!("error: {error}: {message}"),
        }
    }
    if json {
        let object = serde_json::json!({
            "command": command,
            "error": {"code": error.code.as_str(), "message": message},
        });
        println!("{object}");
    }
    ExitCode::from(error.exit_code())
}

/// Runs the command `cli` names, and returns its name, as 02 §2.2 spells it, with its outcome.
fn dispatch(cli: &Cli) -> (&'static str, Result<(), TesseraError>) {
    use schema::{
        AUDIT_SAMPLE, EXECUTION_PLAN, INTEGER_MODEL, LOG_MANIFEST, WEIGHT_COMMITMENT_STATE,
    };
    match &cli.command {
        Command::Compile(a) => (
            "compile",
            pending("compile", &[Input::new(&a.model, &INTEGER_MODEL)], 87),
        ),
        Command::CommitWeights(a) => (
            "commit-weights",
            pending(
                "commit-weights",
                &[
                    Input::new(&a.model, &INTEGER_MODEL),
                    Input::new(&a.plan, &EXECUTION_PLAN),
                ],
                111,
            ),
        ),
        Command::Weights(Weights::Refresh(a)) => (
            "weights refresh",
            pending(
                "weights refresh",
                &[Input::new(&a.weights, &WEIGHT_COMMITMENT_STATE)],
                153,
            ),
        ),
        Command::Prove(a) => (
            "prove",
            pending(
                "prove",
                &[
                    Input::new(&a.plan, &EXECUTION_PLAN),
                    Input::new(&a.model, &INTEGER_MODEL),
                    Input::new(&a.weights, &WEIGHT_COMMITMENT_STATE),
                    Input::new(&a.sample, &AUDIT_SAMPLE),
                ],
                157,
            ),
        ),
        Command::Audit(Audit::Sample(a)) => (
            "audit sample",
            pending(
                "audit sample",
                &[Input::new(&a.manifest, &LOG_MANIFEST)],
                157,
            ),
        ),
        Command::Audit(Audit::Batch(a)) => (
            "audit batch",
            pending("audit batch", &[Input::new(&a.sample, &AUDIT_SAMPLE)], 157),
        ),
        Command::Audit(Audit::Plan(a)) => (
            "audit plan",
            pending("audit plan", &[Input::new(&a.plan, &EXECUTION_PLAN)], 157),
        ),
        Command::Params(Params::Check(_)) => ("params check", pending("params check", &[], 103)),
        Command::Bench(Bench::UnitCosts(_)) => {
            ("bench unit-costs", pending("bench unit-costs", &[], 49))
        }
        Command::Bench(Bench::Model(a)) => ("bench model", bench::model(a, cli.json)),
        Command::Recorder(Recorder::Ref(_)) => ("recorder ref", pending("recorder ref", &[], 139)),
    }
}

/// A TCE artifact a command reads: its path and its schema.
pub(crate) struct Input<'a> {
    path: &'a Path,
    schema: &'static Schema,
}

impl<'a> Input<'a> {
    pub(crate) fn new(path: &'a Path, schema: &'static Schema) -> Self {
        Self { path, schema }
    }

    /// Checks the artifact's envelope in constant memory: its magic, schema, version, lengths
    /// and digest (RFC-0003 'Envelope'), under its schema's limits.
    fn verify(&self) -> Result<(), TesseraError> {
        let shown = self.path.display().to_string();
        let file = File::open(self.path).map_err(io_error(format!("cannot read {shown}")))?;
        let limits = match self.schema.id == schema::INTEGER_MODEL.id {
            true => DecodeLimits::INTEGER_MODEL,
            false => DecodeLimits::MESSAGE,
        };
        match envelope::verify(BufReader::new(file), self.schema, limits) {
            Ok(_) => Ok(()),
            Err(ReadError::Invalid(error)) => Err(TesseraError::from(error).with_source(format!(
                "{shown} is not a valid {} envelope",
                self.schema.name
            ))),
            Err(ReadError::Io(error)) => Err(io_error(format!("cannot read {shown}"))(error)),
        }
    }
}

/// A command of 02 §2.2 that `issue` implements: it checks the envelope of each artifact it
/// reads, then refuses with `ERR-USAGE`.
pub(crate) fn pending(command: &str, inputs: &[Input<'_>], issue: u32) -> Result<(), TesseraError> {
    for input in inputs {
        input.verify()?;
    }
    Err(TesseraError::new(ErrorCode::Usage).with_source(format!(
        "tessera {command} is not implemented yet (#{issue})"
    )))
}

pub(crate) fn io_error(what: String) -> impl FnOnce(io::Error) -> TesseraError {
    move |error| TesseraError::new(ErrorCode::ResIo).with_source(format!("{what}: {error}"))
}

/// Writes `contents` to `path` through a temporary file beside it, renamed on completion, so the
/// path holds either the old file or the whole new one (04 §6).
pub(crate) fn write_atomically(path: &Path, contents: &str) -> Result<(), TesseraError> {
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
