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
//!                         LOGS --transcripts DIR --batch K (--live ENDPOINT | --out PROOF.tnp)
//!                         [--backend {cpu,gpu}] [--devices 0,1,...] [--memory-budget GIB]
//! tessera audit share     --manifest MANIFEST.tlm --policy POLICY.json --commitment HEX --state DIR
//! tessera audit sample    --manifest MANIFEST.tlm --log LOG.tdl
//!                         [--previous-manifest MANIFEST.tlm --previous-log LOG.tdl] [--between MANIFEST.tlm]...
//!                         --state DIR --verifier-share HEX --out SAMPLE.tas
//! tessera audit batch     --sample SAMPLE.tas LOGS --out-dir DIR
//! tessera audit plan      --policy POLICY.json --traffic PROFILE.json (--plan plan.tep | --source-defaults)
//!                         [--eta η] [--target-overhead O] [--out PLAN.json]
//! tessera params check    --params PARAMSET [--lookups N] [--table-rows T]
//! tessera bench unit-costs --device INT --out UNIT_COSTS.json
//! tessera bench model     (--plan plan.tep | --source-defaults) [--context INT,...] [--out MU.json]
//! tessera recorder ref    --listen ADDR --upstream ADDR --log-dir DIR --key FILE --period SECONDS
//! tessera inspect         FILE
//! ```
//!
//! `LOGS` is `audit sample`'s `--manifest` … `--between` flags. `tessera prove` produces one
//! exact-greedy CPU proof in NI or directory Live mode, of a TokenFile (`--reference-tokens`) or
//! of a sample's batch of one request (`--sample`); batches of several requests wait on #357.
//! `tessera compile` runs in `compile`, `tessera commit-weights` in `weights`, `tessera audit
//! share`, `sample`, `batch` and `plan` in `audit`, `tessera bench model`
//! in `bench`, `tessera params check` in `params`, `tessera recorder ref` in `recorder`, and
//! `tessera inspect` in `inspect`. Other commands check their
//! arguments and the envelope of each TCE artifact it reads (RFC-0003), then exits with
//! `ERR-USAGE`, naming the issue that implements it.
//!
//! Every command prints its progress to stderr, readable or, with `--log-format json`, as JSON
//! lines (05 §2, the subscriber of `log`). With `--json` it prints one JSON object to
//! stdout describing its result, or `{"command", "error": {"code", "message"}}` when it fails
//! (02 §2). The exit status is 02 §2.4's (04 §3, RFC-0051): 0 on success, 10 when a soundness
//! target is missed (`tessera params check`), 64 for a usage error, 65 for a malformed artifact,
//! 20 when an artifact cannot be read or written, 70 for an internal error. Errors go to stderr
//! as `error: ERR-...: what happened`; messages hold public data only (05 §3).
//!
//! Every command's outcome leaves through [`run`]'s one exit point, where the run manifest of
//! 05 §6 will be written (#95).

#![forbid(unsafe_code)]
// A command line writes its results to stdout and its errors to stderr (02 §2).
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[cfg(all(feature = "test-rng", not(debug_assertions)))]
compile_error!(
    "the `test-rng` feature pins masking randomness to a seed; it must not be enabled in a release build (RFC-0001 'Cargo features')"
);

mod audit;
mod bench;
mod compile;
mod inspect;
mod log;
mod manifest;
mod metrics;
#[cfg(feature = "otel")]
mod otel;
mod params;
mod prove;
mod recorder;
mod weights;

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Read};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};
use tessera_codec::envelope::{self, ReadError};
use tessera_codec::error::{ErrorCode, TesseraError};
use tessera_codec::schema::{self, Schema};
use tessera_codec::{Decode, DecodeLimits, Public, SecretBuf, event};

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
    /// The format of stderr's lines: readable, or one JSON object per line (05 §2).
    #[arg(
        long,
        global = true,
        value_enum,
        default_value_t,
        value_name = "FORMAT"
    )]
    log_format: log::LogFormat,
    /// The most detailed events written to stderr; `trace` exists in debug builds only.
    #[arg(long, global = true, value_enum, default_value_t, value_name = "LEVEL")]
    log_level: log::LogLevel,
    /// Where `compile`, `commit-weights`, `prove` and `bench model` write their run manifest
    /// (05 §6), instead of their output's directory.
    #[arg(long, global = true, value_name = "PATH")]
    run_manifest: Option<PathBuf>,
    /// Expose the private metrics registry at this explicit HTTP socket (05 §4).
    #[arg(long, global = true, value_name = "IP:PORT")]
    metrics_addr: Option<SocketAddr>,
    /// Export traces to the collector's full HTTP(S) /v1/traces URL (05 §5).
    #[cfg(feature = "otel")]
    #[arg(long, global = true, value_name = "URL")]
    otlp_endpoint: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compiles D into its claim list, execution plan, tables and commitment report.
    Compile(compile::CompileArgs),
    /// Commits D's weights under an execution plan (RFC-0012).
    CommitWeights(weights::CommitWeightsArgs),
    /// Weight-commitment maintenance (RFC-0012).
    #[command(subcommand)]
    Weights(Weights),
    /// Proves a sample of recorded requests, live or non-interactively (RFC-0014).
    Prove(Box<prove::ProveArgs>),
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
    /// Renders a TCE artifact as text (RFC-0003).
    Inspect(inspect::InspectArgs),
}

#[derive(Debug, Subcommand)]
enum Weights {
    /// Re-masks a weight commitment and emits the proof linking it to the old root.
    Refresh(RefreshArgs),
}

#[derive(Debug, Subcommand)]
enum Audit {
    /// Answers a verifier's commitment over a manifest with the prover's share, once per manifest.
    Share(ShareArgs),
    /// Draws a period's audit sample from its log, the previous period's and the coin toss.
    Sample(SampleArgs),
    /// Groups a sample's requests into proof batches.
    Batch(BatchArgs),
    /// Reports an audit policy's detection probability and overhead.
    Plan(PlanArgs),
}

#[derive(Debug, Subcommand)]
enum Params {
    /// Reports a parameter set's soundness, every row of RFC-0014's table.
    Check(params::CheckArgs),
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

/// `tessera weights refresh`.
#[derive(Debug, Args)]
struct RefreshArgs {
    #[arg(long, value_name = "WEIGHTS.twc")]
    weights: PathBuf,
    #[arg(long, value_name = "WEIGHTS2.twc")]
    out: PathBuf,
}

/// `tessera audit share`.
#[derive(Debug, Args)]
struct ShareArgs {
    /// The manifest the coin toss is over.
    #[arg(long, value_name = "MANIFEST.tlm")]
    manifest: PathBuf,
    /// The audit policy the parties fixed before the toss, a `tessera.audit-policy/1` file.
    #[arg(long, value_name = "POLICY.json")]
    policy: PathBuf,
    /// The verifier's commitment `c_V`, 64 hexadecimal digits.
    #[arg(long, value_name = "HEX", value_parser = digest)]
    commitment: [u8; 32],
    /// Where the prover keeps its answers, one file per manifest; every responder of one prover
    /// shares it.
    #[arg(long, value_name = "DIR")]
    state: PathBuf,
}

/// The period's logs, as `tessera audit sample` and `audit batch` read them.
#[derive(Debug, Args)]
struct LogArgs {
    /// The period's manifest.
    #[arg(long, value_name = "MANIFEST.tlm")]
    manifest: PathBuf,
    /// The period's log, its entries' TCE encodings concatenated (RFC-0062 'On disk').
    #[arg(long, value_name = "LOG.tdl")]
    log: PathBuf,
    /// The manifest of the last period of non-zero length before it, unless it has none.
    #[arg(long, value_name = "MANIFEST.tlm", requires = "previous_log")]
    previous_manifest: Option<PathBuf>,
    /// That period's log.
    #[arg(long, value_name = "LOG.tdl", requires = "previous_manifest")]
    previous_log: Option<PathBuf>,
    /// A period of zero length between the two, in order; repeated for each.
    #[arg(long, value_name = "MANIFEST.tlm")]
    between: Vec<PathBuf>,
}

impl LogArgs {
    fn logs(&self) -> audit::Logs<'_> {
        audit::Logs {
            manifest: &self.manifest,
            log: &self.log,
            previous_manifest: self.previous_manifest.as_deref(),
            previous_log: self.previous_log.as_deref(),
            between: &self.between,
        }
    }
}

/// `tessera audit sample`.
#[derive(Debug, Args)]
struct SampleArgs {
    #[command(flatten)]
    logs: LogArgs,
    /// Where `tessera audit share` kept its answer for the manifest: the commitment, the share
    /// and the policy of the toss.
    #[arg(long, value_name = "DIR")]
    state: PathBuf,
    /// The verifier's revealed share `s_V`, 64 hexadecimal digits.
    #[arg(long, value_name = "HEX", value_parser = digest)]
    verifier_share: [u8; 32],
    #[arg(long, value_name = "SAMPLE.tas")]
    out: PathBuf,
}

/// `tessera audit batch`.
#[derive(Debug, Args)]
struct BatchArgs {
    /// The sample, whose policy holds the batch bounds.
    #[arg(long, value_name = "SAMPLE.tas")]
    sample: PathBuf,
    #[command(flatten)]
    logs: LogArgs,
    /// Where the tessera.audit-batches/1 report goes, as batches.json.
    #[arg(long, value_name = "DIR")]
    out_dir: PathBuf,
}

/// `tessera audit plan`.
#[derive(Debug, Args)]
#[command(group(clap::ArgGroup::new("counts").required(true).args(["plan", "source_defaults"])))]
struct PlanArgs {
    /// The audit policy, a `tessera.audit-policy/1` file.
    #[arg(long, value_name = "POLICY.json")]
    policy: PathBuf,
    /// The requests per bucket, their mean length and the violating ones, a
    /// `tessera.audit-traffic/1` file.
    #[arg(long, value_name = "PROFILE.json")]
    traffic: PathBuf,
    /// The execution plan whose counts give the cost model's `μ`.
    #[arg(long, value_name = "plan.tep")]
    plan: Option<PathBuf>,
    /// Use the source's counts and unit costs for `μ` (source §5).
    #[arg(long)]
    source_defaults: bool,
    /// GPUs that prove a request.
    #[arg(long, value_name = "G", default_value_t = tessera_bench::Scenario::SOURCE.gpus)]
    gpus: u32,
    /// Tokens of a proof batch, which share its weight pass.
    #[arg(long, value_name = "T", default_value_t = tessera_bench::Scenario::SOURCE.batch_tokens)]
    batch_tokens: u32,
    /// Ratio of ideal prefill time to serving time per token.
    #[arg(long, value_name = "η", default_value_t = tessera_bench::Scenario::SOURCE.eta)]
    eta: f64,
    /// Amortized overhead the target rates are set for, as a fraction.
    #[arg(long, value_name = "O", default_value_t = tessera_bench::Scenario::SOURCE.target_overhead)]
    target_overhead: f64,
    /// Write the tessera.audit-plan/1 report to this file.
    #[arg(long, value_name = "PLAN.json")]
    out: Option<PathBuf>,
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
    #[arg(long, value_name = "ADDR")]
    upstream: SocketAddr,
    #[arg(long, value_name = "DIR")]
    log_dir: PathBuf,
    /// The recorder's 32-byte Ed25519 seed.
    #[arg(long, value_name = "FILE")]
    key: PathBuf,
    /// The length of a period, in seconds.
    #[arg(long, value_name = "SECONDS")]
    period: u64,
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

/// 32 bytes as 64 hexadecimal digits.
fn digest(text: &str) -> Result<[u8; 32], String> {
    let mut out = [0u8; 32];
    if text.len() != 64 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("not 64 hexadecimal digits".to_owned());
    }
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)
            .map_err(|_| "not 64 hexadecimal digits".to_owned())?;
    }
    Ok(out)
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
    #[cfg(feature = "otel")]
    let endpoint = cli.otlp_endpoint.as_deref();
    #[cfg(not(feature = "otel"))]
    let endpoint = None;
    let _logging = match log::install(cli.log_format, cli.log_level, endpoint) {
        Ok(guard) => guard,
        Err(error) => {
            let _fallback = log::install(cli.log_format, cli.log_level, None);
            return fail(
                None,
                cli.json,
                &error,
                "cannot initialize trace export",
                true,
            );
        }
    };
    tessera_codec::metrics::reset();
    let _metrics = match cli.metrics_addr.map(metrics::Server::start).transpose() {
        Ok(server) => server,
        Err(_) => {
            return fail(
                None,
                cli.json,
                &TesseraError::new(ErrorCode::ResIo),
                "cannot start metrics listener",
                true,
            );
        }
    };
    let started = (std::time::SystemTime::now(), std::time::Instant::now());
    let (name, mut outcome) = dispatch(&cli);
    // 05 §6: the commands that write a run manifest, at their one exit point.
    if ["compile", "commit-weights", "prove", "bench model"].contains(&name)
        && let Some(path) = manifest::path(cli.run_manifest.as_deref())
    {
        let arguments: Vec<String> = args
            .iter()
            .skip(1)
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let rendered = manifest::render(
            name,
            &arguments,
            started.0,
            started.1.elapsed(),
            outcome.as_ref().err(),
        );
        if let Err(error) = write_atomically(&path, &format!("{rendered:#}\n"))
            && outcome.is_ok()
        {
            outcome = Err(error);
        }
    }
    match outcome {
        Ok(code) => code,
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
    // Before parsing succeeds no subscriber is installed: clap has printed its own message.
    if shown {
        match message.is_empty() {
            true => event!(ERROR, "cli.command_failed", error = Public(error); "{error}"),
            false => event!(
                ERROR,
                "cli.command_failed",
                error = Public(error),
                detail = Public(message);
                "{error}: {detail}"
            ),
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
fn dispatch(cli: &Cli) -> (&'static str, Result<ExitCode, TesseraError>) {
    use schema::WEIGHT_COMMITMENT_STATE;
    match &cli.command {
        Command::Compile(a) => (
            "compile",
            compile::compile(a, cli.json).map(|()| ExitCode::SUCCESS),
        ),
        Command::CommitWeights(a) => (
            "commit-weights",
            weights::commit_weights(a, cli.json).map(|()| ExitCode::SUCCESS),
        ),
        Command::Weights(Weights::Refresh(a)) => (
            "weights refresh",
            pending(
                "weights refresh",
                &[Input::new(&a.weights, &WEIGHT_COMMITMENT_STATE)],
                153,
            )
            .map(|()| ExitCode::SUCCESS),
        ),
        Command::Prove(a) => (
            "prove",
            prove::prove(a, cli.json).map(|()| ExitCode::SUCCESS),
        ),
        Command::Audit(Audit::Share(a)) => (
            "audit share",
            audit::share(
                &audit::ShareOptions {
                    manifest: &a.manifest,
                    policy: &a.policy,
                    commitment: a.commitment,
                    state: &a.state,
                },
                cli.json,
            )
            .map(|()| ExitCode::SUCCESS),
        ),
        Command::Audit(Audit::Sample(a)) => (
            "audit sample",
            audit::sample(&audit::SampleOptions {
                logs: a.logs.logs(),
                state: &a.state,
                verifier_share: a.verifier_share,
                out: &a.out,
            })
            .map(|()| ExitCode::SUCCESS),
        ),
        Command::Audit(Audit::Batch(a)) => (
            "audit batch",
            audit::batch(
                &audit::BatchOptions {
                    sample: &a.sample,
                    logs: a.logs.logs(),
                    out_dir: &a.out_dir,
                },
                cli.json,
            )
            .map(|()| ExitCode::SUCCESS),
        ),
        Command::Audit(Audit::Plan(a)) => (
            "audit plan",
            audit::plan(
                &audit::PlanOptions {
                    policy: &a.policy,
                    traffic: &a.traffic,
                    plan: a.plan.as_deref(),
                    gpus: a.gpus,
                    batch_tokens: a.batch_tokens,
                    eta: a.eta,
                    target_overhead: a.target_overhead,
                    out: a.out.as_deref(),
                },
                cli.json,
            )
            .map(|()| ExitCode::SUCCESS),
        ),
        Command::Params(Params::Check(a)) => ("params check", params::check(a, cli.json)),
        Command::Bench(Bench::UnitCosts(_)) => (
            "bench unit-costs",
            pending("bench unit-costs", &[], 49).map(|()| ExitCode::SUCCESS),
        ),
        Command::Bench(Bench::Model(a)) => (
            "bench model",
            bench::model(a, cli.json).map(|()| ExitCode::SUCCESS),
        ),
        Command::Recorder(Recorder::Ref(a)) => (
            "recorder ref",
            recorder::run(&recorder::Options {
                listen: a.listen,
                upstream: a.upstream,
                log_dir: &a.log_dir,
                key: &a.key,
                period: std::time::Duration::from_secs(a.period),
            })
            .map(|()| ExitCode::SUCCESS),
        ),
        Command::Inspect(a) => (
            "inspect",
            inspect::inspect(a, cli.json).map(|()| ExitCode::SUCCESS),
        ),
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
        if !fs::metadata(self.path)
            .map_err(io_error(format!("cannot read {shown}")))?
            .is_file()
        {
            return Err(TesseraError::new(ErrorCode::ResIo));
        }
        let file = File::open(self.path).map_err(io_error(format!("cannot read {shown}")))?;
        let limits = match self.schema.id == schema::INTEGER_MODEL.id {
            true => DecodeLimits::INTEGER_MODEL,
            false => DecodeLimits::MESSAGE,
        };
        // OBS-6: the envelope reader clears its buffer; read-ahead would add an uncleared copy.
        match envelope::verify(file, self.schema, limits) {
            Ok(_) => {
                manifest::input(self.path, self.schema);
                Ok(())
            }
            Err(ReadError::Invalid(error)) => Err(TesseraError::from(error).with_source(format!(
                "{shown} is not a valid {} envelope",
                self.schema.name
            ))),
            Err(ReadError::Io(error)) => Err(io_error(format!("cannot read {shown}"))(error)),
        }
    }

    /// Opens the artifact's envelope and decodes its payload under its schema's limits
    /// (RFC-0003 'Envelope').
    pub(crate) fn open<T: Decode>(&self) -> Result<T, TesseraError> {
        self.verify()?;
        let shown = self.path.display().to_string();
        let limits = match self.schema.id == schema::INTEGER_MODEL.id {
            true => DecodeLimits::INTEGER_MODEL,
            false => DecodeLimits::MESSAGE,
        };
        let mut file = File::open(self.path).map_err(io_error(format!("cannot read {shown}")))?;
        let metadata = file
            .metadata()
            .map_err(io_error(format!("cannot read {shown}")))?;
        if !metadata.is_file() {
            return Err(TesseraError::new(ErrorCode::ResIo));
        }
        if metadata.len() > limits.max_total_bytes {
            return Err(TesseraError::new(ErrorCode::CodecLimit));
        }
        let len =
            usize::try_from(metadata.len()).map_err(|_| TesseraError::new(ErrorCode::ResMemory))?;
        let mut bytes = SecretBuf::try_with_capacity(len)
            .map_err(|_| TesseraError::new(ErrorCode::ResMemory))?;
        bytes.resize(len, 0);
        file.read_exact(&mut bytes)
            .map_err(io_error(format!("cannot read {shown}")))?;
        let mut extra =
            SecretBuf::try_with_capacity(1).map_err(|_| TesseraError::new(ErrorCode::ResMemory))?;
        extra.push(0);
        if file
            .read(&mut extra)
            .map_err(io_error(format!("cannot read {shown}")))?
            != 0
        {
            return Err(TesseraError::new(ErrorCode::ResIo));
        }
        envelope::open(&bytes, self.schema, limits).map_err(|error| {
            TesseraError::from(error).with_source(format!(
                "{shown} is not a valid {} envelope",
                self.schema.name
            ))
        })
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
    write_bytes_atomically(path, contents.as_bytes())
}

/// Writes `bytes` to `path` through a temporary file beside it, renamed on completion (04 §6).
pub(crate) fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), TesseraError> {
    let shown = path.display().to_string();
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(".tmp-{}", std::process::id()));
    let temporary = PathBuf::from(temporary);
    fs::write(&temporary, bytes).map_err(io_error(format!("cannot write {shown}")))?;
    fs::rename(&temporary, path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        io_error(format!("cannot write {shown}"))(error)
    })?;
    manifest::output_envelope(path, bytes);
    Ok(())
}
