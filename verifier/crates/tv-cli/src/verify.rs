// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol/1 CLI policy: authenticate the entire statement with the caller's digest.
use crate::observability::{Observation, nanos};
use crate::{
    CommandOutput, files,
    options::{self, Options},
};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tv_codec::{DecodeLimits, Envelope, WeightCommitmentPublic};
use tv_core::driver::CheckEvent;
use tv_core::{
    ResourceLimits, VerifierConfig,
    claims::ClaimList,
    params::ParamSet,
    protocol::{FixedBits, NiProof},
    verdict::{ErrorCode, RejectReason, Verdict, VerifierError},
};
use tv_hash::{Domain, hash};

pub(crate) const HELP: &str = "\
tessera-verify verify: non-hiding protocol/1, one exact-greedy request

USAGE:
  tessera-verify verify --claims CLAIMS.tcl --weights WEIGHTS.twr
    --weights-root HEX --statement-digest HEX --proof PROOF.tnp --bound BITS
    [--out VERDICT.json] [--json] [--max-artifact-bytes N]
    [--max-memory-bytes N] [--timeout-ms N]
    [--log-format human|json] [--run-manifest PATH]

The statement digest is a trusted caller-owned H(statement,TCE(PublicStatement));
obtain it independently of the proof. --bound is exact bits in increments of1/256.
Input files must be regular files. The report records real CLI phase timings;
core per-check timings remain null; the run manifest records check histograms.
--log-format json writes public span events to stderr; human is the default.
The run manifest defaults beside --out, or beside --proof when --out is absent.
Anchor, ledger and ZK are not verified.
Test-small builds certify nothing. --manifest belongs to future anchor support.
";

struct Config<'a> {
    claims: &'a str,
    weights: &'a str,
    proof: &'a str,
    out: Option<&'a str>,
    run_manifest: Option<&'a str>,
    log_json: bool,
    root: [u8; 32],
    statement: [u8; 32],
    bound: FixedBits,
    limits: ResourceLimits,
}
impl<'a> Config<'a> {
    fn parse(args: &'a [String]) -> Result<Self, ErrorCode> {
        let o = Options::parse(
            args,
            &[
                "--claims",
                "--weights",
                "--proof",
                "--out",
                "--log-format",
                "--run-manifest",
                "--weights-root",
                "--statement-digest",
                "--bound",
                "--max-artifact-bytes",
                "--max-memory-bytes",
                "--timeout-ms",
            ],
        )?;
        let defaults = ResourceLimits::default();
        Ok(Self {
            claims: o.required("--claims")?,
            weights: o.required("--weights")?,
            proof: o.required("--proof")?,
            out: o.get("--out"),
            run_manifest: o.get("--run-manifest"),
            log_json: match o.get("--log-format") {
                None | Some("human") => false,
                Some("json") => true,
                _ => return Err(ErrorCode::Usage),
            },
            root: options::digest(o.required("--weights-root")?)?,
            statement: options::digest(o.required("--statement-digest")?)?,
            bound: FixedBits(options::bound(o.required("--bound")?)?),
            limits: ResourceLimits {
                max_artifact_bytes: o
                    .number("--max-artifact-bytes", defaults.max_artifact_bytes)?,
                max_memory_bytes: o.number("--max-memory-bytes", defaults.max_memory_bytes)?,
                max_wall_time: Duration::from_millis(o.number("--timeout-ms", 300_000)?),
            },
        })
    }
}
pub(crate) struct Execution {
    start: Instant,
    durations: [Option<u64>; 3],
    outcomes: [&'static str; 3],
    pub(crate) pinned: bool,
    observation: Observation,
}
impl Execution {
    pub(crate) fn new() -> Self {
        Self {
            start: Instant::now(),
            durations: [None; 3],
            outcomes: ["error"; 3],
            pinned: false,
            observation: Observation::new(),
        }
    }
    pub(crate) fn phase<T>(
        &mut self,
        index: usize,
        f: impl FnOnce() -> Result<T, ErrorCode>,
    ) -> Result<T, ErrorCode> {
        let kind = ["load", "statement_pin", "verification"][index];
        self.observation.record(CheckEvent::Started { kind });
        let at = Instant::now();
        let result = f();
        let duration = at.elapsed();
        self.observation.record(CheckEvent::Finished {
            kind,
            duration,
            completed: result.is_ok(),
        });
        self.durations[index] = Some(nanos(duration));
        self.outcomes[index] = if result.is_ok() { "completed" } else { "error" };
        result
    }
    pub(crate) fn remaining(&self, limit: Duration) -> Result<Duration, ErrorCode> {
        limit
            .checked_sub(self.start.elapsed())
            .filter(|d| !d.is_zero())
            .ok_or(ErrorCode::Time)
    }
    pub(crate) fn evaluate(&mut self, f: impl FnOnce() -> Verdict) -> Verdict {
        self.evaluate_observed(|_| f())
    }
    fn evaluate_observed(&mut self, f: impl FnOnce(&mut Observation) -> Verdict) -> Verdict {
        self.observation.record(CheckEvent::Started {
            kind: "verification",
        });
        let at = Instant::now();
        let verdict = f(&mut self.observation);
        let duration = at.elapsed();
        self.observation.record(CheckEvent::Finished {
            kind: "verification",
            duration,
            completed: matches!(verdict, Verdict::Accept(_)),
        });
        self.durations[2] = Some(nanos(duration));
        self.outcomes[2] = match &verdict {
            Verdict::Accept(_) => "accept",
            Verdict::Reject(_) => "reject",
            Verdict::Inconclusive(_) => "inconclusive",
        };
        verdict
    }
}
pub(crate) fn inconclusive(code: ErrorCode) -> Verdict {
    Verdict::Inconclusive(VerifierError { code })
}

pub(crate) fn decode_limits(limits: ResourceLimits) -> DecodeLimits {
    DecodeLimits {
        max_total_bytes: limits.max_artifact_bytes.min(limits.max_memory_bytes / 8),
        max_vec_len: DecodeLimits::MESSAGE
            .max_vec_len
            .min(limits.max_memory_bytes / 8),
        ..DecodeLimits::MESSAGE
    }
}
pub(crate) fn public_inputs(
    claims: &str,
    weights: &str,
    input: &mut files::Inputs,
    limits: &DecodeLimits,
) -> Result<(ClaimList, WeightCommitmentPublic), ErrorCode> {
    let claims = {
        let bytes = input.read(Path::new(claims))?;
        ClaimList::open(&bytes, limits).map_err(|e| ErrorCode::from(e.code()))?
    };
    let weights = {
        let bytes = input.read(Path::new(weights))?;
        Envelope::open(&bytes, WeightCommitmentPublic::EXPECTED, limits, |b| {
            hash(Domain::Envelope, b)
        })
        .and_then(|e| e.decode_payload(|c, _| WeightCommitmentPublic::decode(c)))
        .map_err(|e| ErrorCode::from(e.code()))?
    };
    Ok((claims, weights))
}

fn verify(c: &Config<'_>, e: &mut Execution) -> Result<Verdict, ErrorCode> {
    e.remaining(c.limits.max_wall_time)?;
    // Reserve a conservative input/decoded-object allowance before any file allocation.
    let mut input = files::Inputs {
        remaining: c.limits.max_memory_bytes / 8,
        max_artifact: c.limits.max_artifact_bytes,
    };
    let limits = decode_limits(c.limits);
    let mut artifacts = Observation::new();
    let loaded = e.phase(0, || {
        let bytes = input.read(Path::new(c.claims))?;
        let claims = ClaimList::open(&bytes, &limits).map_err(|e| ErrorCode::from(e.code()))?;
        artifacts.params = Some(claims.params.as_str().into());
        artifacts.artifact(c.claims, "ClaimList", "public", &bytes);
        let bytes = input.read(Path::new(c.weights))?;
        let weights = Envelope::open(&bytes, WeightCommitmentPublic::EXPECTED, &limits, |b| {
            hash(Domain::Envelope, b)
        })
        .and_then(|e| e.decode_payload(|c, _| WeightCommitmentPublic::decode(c)))
        .map_err(|e| ErrorCode::from(e.code()))?;
        artifacts.artifact(c.weights, "WeightCommitmentPublic", "public", &bytes);
        let bytes = input.read(Path::new(c.proof))?;
        let proof = NiProof::open(&bytes, &limits).map_err(|e| ErrorCode::from(e.code()))?;
        artifacts.artifact(c.proof, "NiProof", "protocol", &bytes);
        Ok((claims, weights, proof))
    });
    e.observation.inputs = artifacts.inputs;
    e.observation.proof_digest = artifacts.proof_digest;
    e.observation.params = artifacts.params;
    let (claims, weights, proof) = loaded?;
    e.remaining(c.limits.max_wall_time)?;
    e.phase(1, || {
        if proof.round1.statement.digest() != c.statement
            || weights.weights_root(|b| hash(Domain::WeightsRoot, b)) != c.root
        {
            return Err(ErrorCode::Usage);
        }
        Ok(())
    })?;
    e.pinned = true;
    let Some(params) = ParamSet::accept(&claims.params) else {
        return Ok(Verdict::Reject(RejectReason::Params {
            name: claims.params,
        }));
    };
    let cfg = VerifierConfig {
        params,
        weights,
        accept_bound: c.bound,
        limits: ResourceLimits {
            max_wall_time: e.remaining(c.limits.max_wall_time)?,
            ..c.limits
        },
    };
    // Only after the external whole-statement pin matches is its content trusted.
    let verdict = e.evaluate_observed(|observation| {
        tv_core::driver::verify_non_interactive_observed(
            &claims,
            &proof.round1.statement,
            &proof,
            &cfg,
            &mut |event| observation.record(event),
        )
    });
    Ok(verdict)
}

fn json(verdict: &Verdict, e: &Execution) -> Result<String, ErrorCode> {
    let mut json = verdict.to_json().map_err(|e| e.code)?;
    if !json.ends_with("}\n") {
        return Err(ErrorCode::Internal);
    }
    json.try_reserve_exact(1024)
        .map_err(|_| ErrorCode::Memory)?;
    json.truncate(json.len() - 2);
    write!(json,",\"execution\":{{\"source\":\"cli\",\"duration_ns\":{},\"statement_pinned\":{},\"exit_code\":{},\"trace\":[",nanos(e.start.elapsed()),e.pinned,verdict.exit_code()).map_err(|_|ErrorCode::Internal)?;
    let mut first = true;
    for (i, name) in ["load", "statement_pin", "verification"].iter().enumerate() {
        if let Some(duration) = e.durations[i] {
            if !first {
                json.push(',');
            }
            first = false;
            write!(
                json,
                "{{\"phase\":\"{name}\",\"outcome\":\"{}\",\"duration_ns\":{duration}}}",
                e.outcomes[i]
            )
            .map_err(|_| ErrorCode::Internal)?;
        }
    }
    json.push_str("]}}\n");
    Ok(json)
}

pub(crate) fn failure(code: ErrorCode) -> CommandOutput {
    let verdict = inconclusive(code);
    let (stdout, exit_code) = match verdict.to_json() {
        Ok(stdout) => (stdout, code.exit_code()),
        Err(error) => (String::new(), error.code.exit_code()),
    };
    CommandOutput {
        exit_code,
        stdout,
        stderr: String::new(),
    }
}
pub(crate) fn run(args: &[String], json_flag: bool) -> CommandOutput {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        return CommandOutput {
            exit_code: 0,
            stdout: HELP.into(),
            stderr: String::new(),
        };
    }
    let config = match Config::parse(args) {
        Ok(c) => c,
        Err(code) => {
            let mut output = crate::command_error(
                "verify",
                "invalid, missing or duplicate option; see verify --help",
                json_flag,
                code.id(),
                code.exit_code(),
            );
            if args
                .windows(2)
                .any(|a| a[0] == "--log-format" && a[1] == "json")
            {
                output.stderr = Observation::new().logs(output.exit_code);
            }
            return output;
        }
    };
    let mut execution = Execution::new();
    let manifest_path = config.run_manifest.map_or_else(
        || Path::new(config.out.unwrap_or(config.proof)).with_file_name("run-manifest.json"),
        PathBuf::from,
    );
    let manifest_path = manifest_path.to_string_lossy();
    let checked_output =
        files::output_distinct(config.out, &[config.claims, config.weights, config.proof])
            .and_then(|()| {
                files::output_distinct(
                    Some(&manifest_path),
                    &[config.claims, config.weights, config.proof],
                )
            })
            .and_then(|()| {
                files::output_distinct(
                    Some(&manifest_path),
                    &config.out.into_iter().collect::<Vec<_>>(),
                )
            });
    let mut verdict = match checked_output {
        Ok(()) => verify(&config, &mut execution).unwrap_or_else(inconclusive),
        Err(code) => inconclusive(code),
    };
    let (mut report, written) = finish_report(&mut verdict, &execution, config.out, checked_output);
    if checked_output.is_ok() {
        let error = match &verdict {
            Verdict::Inconclusive(e) => Some(e.code.id()),
            Verdict::Reject(r) => Some(r.code()),
            Verdict::Accept(_) => None,
        };
        // Only a report successfully written by this invocation is an output artifact.
        let written = config.out.filter(|_| written);
        if let Err(code) = crate::manifest::write(
            Path::new(manifest_path.as_ref()),
            args,
            &execution.observation,
            execution.start.elapsed(),
            report.exit_code,
            error,
            written,
        ) {
            verdict = inconclusive(code);
            report = finish_report(&mut verdict, &execution, config.out, checked_output).0;
        }
    }
    let mut output = render(verdict, report, &execution, json_flag);
    if config.log_json {
        output.stderr = execution.observation.logs(output.exit_code);
    }
    output
}

pub(crate) fn finish(
    mut verdict: Verdict,
    execution: Execution,
    out: Option<&str>,
    json_flag: bool,
    checked_output: Result<(), ErrorCode>,
) -> CommandOutput {
    let (report, _) = finish_report(&mut verdict, &execution, out, checked_output);
    render(verdict, report, &execution, json_flag)
}

fn finish_report(
    verdict: &mut Verdict,
    execution: &Execution,
    out: Option<&str>,
    checked_output: Result<(), ErrorCode>,
) -> (CommandOutput, bool) {
    let mut report = match json(verdict, execution) {
        Ok(report) => report,
        Err(code) => {
            *verdict = inconclusive(code);
            return (failure(code), false);
        }
    };
    let mut written = false;
    if let (Ok(()), Some(out)) = (checked_output, out) {
        match files::write(Path::new(out), report.as_bytes()) {
            Ok(()) => written = true,
            Err(code) => {
                *verdict = inconclusive(code);
                report = match json(verdict, execution) {
                    Ok(report) => report,
                    Err(code) => {
                        *verdict = inconclusive(code);
                        return (failure(code), false);
                    }
                };
            }
        }
    }
    (
        CommandOutput {
            exit_code: verdict.exit_code(),
            stdout: report,
            stderr: String::new(),
        },
        written,
    )
}
fn render(
    verdict: Verdict,
    report: CommandOutput,
    execution: &Execution,
    json_flag: bool,
) -> CommandOutput {
    if json_flag {
        return report;
    }
    let exit_code = report.exit_code;
    let text = match &verdict {
        Verdict::Accept(r) if r.params.name() == "test-small/1" => {
            "Accept (test-small/1; test parameters certify nothing)\n".into()
        }
        Verdict::Accept(_) => "Accept\n".into(),
        Verdict::Reject(r) => format!("Reject ({})\n", r.code()),
        Verdict::Inconclusive(e) => format!("Inconclusive ({})\n", e.code.id()),
    };
    CommandOutput {
        exit_code,
        stdout: text,
        stderr: format!(
            "verify: completed in {} ns; protocol/1, anchor/ledger/ZK unchecked\n",
            nanos(execution.start.elapsed())
        ),
    }
}
