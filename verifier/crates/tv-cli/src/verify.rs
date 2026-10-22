// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol/1 CLI policy: authenticate the entire statement with the caller's digest.
use crate::{
    CommandOutput, files,
    options::{self, Options},
};
use std::fmt::Write;
use std::path::Path;
use std::time::{Duration, Instant};
use tv_codec::{DecodeLimits, Envelope, WeightCommitmentPublic};
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

The statement digest is a trusted caller-owned H(statement,TCE(PublicStatement));
obtain it independently of the proof. --bound is exact bits in increments of1/256.
Input files must be regular files. The report records real CLI phase timings;
core per-check timings remain null. Anchor, ledger and ZK are not verified.
Test-small builds certify nothing. --manifest belongs to future anchor support.
";

struct Config<'a> {
    claims: &'a str,
    weights: &'a str,
    proof: &'a str,
    out: Option<&'a str>,
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
struct Execution {
    start: Instant,
    durations: [Option<u64>; 3],
    outcomes: [&'static str; 3],
    pinned: bool,
}
impl Execution {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            durations: [None; 3],
            outcomes: ["error"; 3],
            pinned: false,
        }
    }
    fn phase<T>(
        &mut self,
        index: usize,
        f: impl FnOnce() -> Result<T, ErrorCode>,
    ) -> Result<T, ErrorCode> {
        let at = Instant::now();
        let result = f();
        self.durations[index] = Some(nanos(at.elapsed()));
        self.outcomes[index] = if result.is_ok() { "completed" } else { "error" };
        result
    }
    fn remaining(&self, limit: Duration) -> Result<Duration, ErrorCode> {
        limit
            .checked_sub(self.start.elapsed())
            .filter(|d| !d.is_zero())
            .ok_or(ErrorCode::Time)
    }
}
fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
pub(crate) fn inconclusive(code: ErrorCode) -> Verdict {
    Verdict::Inconclusive(VerifierError { code })
}

fn verify(c: &Config<'_>, e: &mut Execution) -> Result<Verdict, ErrorCode> {
    e.remaining(c.limits.max_wall_time)?;
    // Reserve a conservative input/decoded-object allowance before any file allocation.
    let mut input = files::Inputs {
        remaining: c.limits.max_memory_bytes / 8,
        max_artifact: c.limits.max_artifact_bytes,
    };
    let limits = DecodeLimits {
        max_total_bytes: c
            .limits
            .max_artifact_bytes
            .min(c.limits.max_memory_bytes / 8),
        max_vec_len: DecodeLimits::MESSAGE
            .max_vec_len
            .min(c.limits.max_memory_bytes / 8),
        ..DecodeLimits::MESSAGE
    };
    let (claims, weights, proof) = e.phase(0, || {
        let claims = {
            let bytes = input.read(Path::new(c.claims))?;
            ClaimList::open(&bytes, &limits).map_err(|e| ErrorCode::from(e.code()))?
        };
        let weights = {
            let bytes = input.read(Path::new(c.weights))?;
            Envelope::open(&bytes, WeightCommitmentPublic::EXPECTED, &limits, |b| {
                hash(Domain::Envelope, b)
            })
            .and_then(|e| e.decode_payload(|c, _| WeightCommitmentPublic::decode(c)))
            .map_err(|e| ErrorCode::from(e.code()))?
        };
        let proof = {
            let bytes = input.read(Path::new(c.proof))?;
            NiProof::open(&bytes, &limits).map_err(|e| ErrorCode::from(e.code()))?
        };
        Ok((claims, weights, proof))
    })?;
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
    let at = Instant::now();
    // Only after the external whole-statement pin matches is its content trusted.
    let verdict = tv_core::verify_non_interactive(&claims, &proof.round1.statement, &proof, &cfg);
    e.durations[2] = Some(nanos(at.elapsed()));
    e.outcomes[2] = match &verdict {
        Verdict::Accept(_) => "accept",
        Verdict::Reject(_) => "reject",
        Verdict::Inconclusive(_) => "inconclusive",
    };
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
            return crate::command_error(
                "verify",
                "invalid, missing or duplicate option; see verify --help",
                json_flag,
                code.id(),
                code.exit_code(),
            );
        }
    };
    let mut execution = Execution::new();
    let checked_output =
        files::output_distinct(config.out, &[config.claims, config.weights, config.proof]);
    let mut verdict = match checked_output {
        Ok(()) => verify(&config, &mut execution).unwrap_or_else(inconclusive),
        Err(code) => inconclusive(code),
    };
    let mut report = match json(&verdict, &execution) {
        Ok(report) => report,
        Err(code) => return failure(code),
    };
    if let (Ok(()), Some(out)) = (checked_output, config.out)
        && let Err(code) = files::write(Path::new(out), report.as_bytes())
    {
        verdict = inconclusive(code);
        report = match json(&verdict, &execution) {
            Ok(report) => report,
            Err(code) => return failure(code),
        };
    }
    let exit_code = verdict.exit_code();
    if json_flag {
        return CommandOutput {
            exit_code,
            stdout: report,
            stderr: String::new(),
        };
    }
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
