// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol/1 CLI policy: authenticate the entire statement with the caller's digest.
//!
//! The CLI reads and decodes the inputs, authenticates the pins and selects the parameter set
//! before tv-core runs: that work is RFC-0019's step 1, which tv-core's own step 1 continues
//! ([`Trail`] joins the two), and a parameter set the verifier does not accept is step 2's
//! rejection. Each check is reported to the trail as it completes (05 §2, §5).
use crate::{
    CommandOutput, files,
    log::{Flags, Trail},
    manifest,
    options::{self, Options},
};
use std::fmt::Write;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};
use tv_codec::{Cursor, DecodeError, DecodeLimits, Envelope, Expected, WeightCommitmentPublic};
use tv_core::{
    ResourceLimits, VerifierConfig,
    claims::{self, ClaimList},
    observe::{CheckKind, CheckOutcome, Observer},
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
    [--max-memory-bytes N] [--timeout-ms N] [--run-manifest PATH]
    [--log-format human|json] [--log-level LEVEL]

The statement digest is a trusted caller-owned H(statement,TCE(PublicStatement));
obtain it independently of the proof. --bound is exact bits in increments of1/256.
Input files must be regular files. The report records real CLI phase timings;
core per-check timings remain null. Anchor, ledger and ZK are not verified.
Test-small builds certify nothing. --manifest belongs to future anchor support.
--run-manifest writes a public tessera.run-manifest/1 at PATH after the report; PATH must
differ from the inputs and --out. --log-level debug logs one line per check.
";

struct Config<'a> {
    claims: &'a str,
    weights: &'a str,
    proof: &'a str,
    out: Option<&'a str>,
    manifest: Option<&'a str>,
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
                "--run-manifest",
            ],
        )?;
        let defaults = ResourceLimits::default();
        Ok(Self {
            claims: o.required("--claims")?,
            weights: o.required("--weights")?,
            proof: o.required("--proof")?,
            out: o.get("--out"),
            manifest: o.get("--run-manifest"),
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
}
impl Execution {
    pub(crate) fn new() -> Self {
        Self {
            start: Instant::now(),
            durations: [None; 3],
            outcomes: ["error"; 3],
            pinned: false,
        }
    }
    pub(crate) fn phase<T>(
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
    pub(crate) fn remaining(&self, limit: Duration) -> Result<Duration, ErrorCode> {
        limit
            .checked_sub(self.start.elapsed())
            .filter(|d| !d.is_zero())
            .ok_or(ErrorCode::Time)
    }
    pub(crate) fn evaluate(&mut self, f: impl FnOnce() -> Verdict) -> Verdict {
        let at = Instant::now();
        let verdict = f();
        self.durations[2] = Some(nanos(at.elapsed()));
        self.outcomes[2] = match &verdict {
            Verdict::Accept(_) => "accept",
            Verdict::Reject(_) => "reject",
            Verdict::Inconclusive(_) => "inconclusive",
        };
        verdict
    }
}
fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
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
/// What the run manifest names of the inputs (05 §6): each envelope's digest once it verifies,
/// and the claim list's parameter set once it decodes.
#[derive(Default)]
pub(crate) struct Opened {
    pub(crate) claims: Option<[u8; 32]>,
    pub(crate) weights: Option<[u8; 32]>,
    pub(crate) proof: Option<[u8; 32]>,
    pub(crate) params: Option<String>,
}

/// Opens `bytes` as an envelope of `expected`, recording its digest once it verifies, then
/// decodes its payload: each artifact's own `open`, in RFC-0003's order, in two steps so that an
/// artifact whose payload fails is still named by its digest.
fn open<T>(
    bytes: &[u8],
    expected: Expected,
    limits: &DecodeLimits,
    digest: &mut Option<[u8; 32]>,
    decode: impl FnOnce(&mut Cursor<'_>) -> Result<T, DecodeError>,
) -> Result<T, ErrorCode> {
    let mut computed = None;
    let envelope = Envelope::open(bytes, expected, limits, |b| {
        let d = hash(Domain::Envelope, b);
        computed = Some(d);
        d
    })
    .map_err(|e| ErrorCode::from(e.code()))?;
    *digest = computed;
    envelope
        .decode_payload(|c, _| decode(c))
        .map_err(|e| ErrorCode::from(e.code()))
}

pub(crate) fn public_inputs(
    claims: &str,
    weights: &str,
    input: &mut files::Inputs,
    limits: &DecodeLimits,
    opened: &mut Opened,
) -> Result<(ClaimList, WeightCommitmentPublic), ErrorCode> {
    let claims = {
        let bytes = input.read(Path::new(claims))?;
        open(&bytes, claims::SCHEMA, limits, &mut opened.claims, |c| {
            ClaimList::decode(c)
        })?
    };
    opened.params = Some(claims.params.as_str().to_owned());
    let weights = {
        let bytes = input.read(Path::new(weights))?;
        open(
            &bytes,
            WeightCommitmentPublic::EXPECTED,
            limits,
            &mut opened.weights,
            WeightCommitmentPublic::decode,
        )?
    };
    Ok((claims, weights))
}

/// The CLI's part of step 1: the time limit, the inputs read and decoded, and the pins.
fn load(
    c: &Config<'_>,
    e: &mut Execution,
    trail: &mut Trail,
    opened: &mut Opened,
) -> Result<(ClaimList, WeightCommitmentPublic, NiProof), ErrorCode> {
    e.remaining(c.limits.max_wall_time)?;
    // Reserve a conservative input/decoded-object allowance before any file allocation.
    let mut input = files::Inputs {
        remaining: c.limits.max_memory_bytes / 8,
        max_artifact: c.limits.max_artifact_bytes,
    };
    let limits = decode_limits(c.limits);
    let loaded = e.phase(0, || {
        let (claims, weights) = public_inputs(c.claims, c.weights, &mut input, &limits, opened)?;
        let proof = {
            let bytes = input.read(Path::new(c.proof))?;
            open(
                &bytes,
                NiProof::SCHEMA,
                &limits,
                &mut opened.proof,
                NiProof::decode,
            )?
        };
        Ok((claims, weights, proof))
    });
    // `verify{proof_digest}` records the digest once the envelope verifies (05 §5).
    if let Some(digest) = &opened.proof {
        trail.record_proof(digest);
    }
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
    Ok((claims, weights, proof))
}

fn verify(
    c: &Config<'_>,
    e: &mut Execution,
    trail: &mut Trail,
    opened: &mut Opened,
) -> Result<Verdict, ErrorCode> {
    let input = Instant::now();
    let failed = |trail: &mut Trail, code| {
        trail.check(
            CheckKind::Input,
            CheckOutcome::Inconclusive,
            input.elapsed(),
        );
        code
    };
    let (claims, weights, proof) = load(c, e, trail, opened).map_err(|code| failed(trail, code))?;
    let selected = Instant::now();
    let Some(params) = ParamSet::accept(&claims.params) else {
        trail.check(
            CheckKind::Input,
            CheckOutcome::Pass,
            selected.duration_since(input),
        );
        trail.check(CheckKind::Params, CheckOutcome::Reject, selected.elapsed());
        return Ok(Verdict::Reject(RejectReason::Params {
            name: claims.params,
        }));
    };
    let remaining = e
        .remaining(c.limits.max_wall_time)
        .map_err(|code| failed(trail, code))?;
    let cfg = VerifierConfig {
        params,
        weights,
        accept_bound: c.bound,
        limits: ResourceLimits {
            max_wall_time: remaining,
            ..c.limits
        },
    };
    trail.check(CheckKind::Input, CheckOutcome::Pass, input.elapsed());
    // Only after the external whole-statement pin matches is its content trusted.
    let verdict = e.evaluate(|| {
        let observer: &mut dyn Observer = trail;
        tv_core::verify_non_interactive_observed(
            &claims,
            &proof.round1.statement,
            &proof,
            &cfg,
            observer,
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
pub(crate) fn run(args: &[String], flags: Flags, command_line: &[String]) -> CommandOutput {
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
                flags,
                code.id(),
                code.exit_code(),
            );
        }
    };
    let mut execution = Execution::new();
    let started_at = SystemTime::now();
    let mut trail = Trail::new(flags);
    let inputs = [config.claims, config.weights, config.proof];
    // Neither output may name an input, nor the manifest the report, before verification.
    let checked_output = files::output_distinct(config.out, &inputs).and_then(|()| {
        config.manifest.map_or(Ok(()), |path| {
            files::manifest_distinct(path, &inputs, config.out)
        })
    });
    let mut opened = Opened::default();
    let verdict = match checked_output {
        Ok(()) => {
            verify(&config, &mut execution, &mut trail, &mut opened).unwrap_or_else(inconclusive)
        }
        Err(code) => inconclusive(code),
    };
    let manifest = config.manifest.map(|path| Manifest {
        path,
        arguments: command_line,
        started_at,
        opened: &opened,
        inputs,
    });
    finish(
        verdict,
        execution,
        config.out,
        flags,
        checked_output,
        trail,
        manifest,
    )
}

/// What `finish` needs to write the run manifest at `path` (05 §6).
pub(crate) struct Manifest<'a> {
    path: &'a str,
    arguments: &'a [String],
    started_at: SystemTime,
    opened: &'a Opened,
    /// The claim list's, the weight commitment's and the proof's paths.
    inputs: [&'a str; 3],
}

impl Manifest<'_> {
    fn render(&self, verdict: &Verdict, report: Option<&str>, run: (&Execution, &Trail)) -> String {
        let (execution, trail) = run;
        let [claims, weights, proof] = self.inputs;
        let inputs = [
            manifest::Artifact {
                path: claims,
                schema: "ClaimList",
                class: "public",
                digest: self.opened.claims,
            },
            manifest::Artifact {
                path: weights,
                schema: "WeightCommitmentPublic",
                class: "public",
                digest: self.opened.weights,
            },
            manifest::Artifact {
                path: proof,
                schema: "NiProof",
                class: "protocol",
                digest: self.opened.proof,
            },
        ];
        let outputs: Vec<manifest::Artifact<'_>> = report
            .map(|path| manifest::Artifact {
                path,
                schema: "tessera.verdict/1",
                class: "public",
                digest: None,
            })
            .into_iter()
            .collect();
        manifest::render(&manifest::Run {
            arguments: self.arguments,
            started_at: self.started_at,
            params: self.opened.params.as_deref(),
            inputs: &inputs,
            outputs: &outputs,
            checks: trail.checks(),
            elapsed: execution.start.elapsed(),
            exit_code: verdict.exit_code(),
            error_code: match verdict {
                Verdict::Accept(_) => None,
                Verdict::Reject(reason) => Some(reason.code()),
                Verdict::Inconclusive(error) => Some(error.code.id()),
            },
        })
    }
}

/// Writes the report, then the run manifest when asked, then ends the `verify` span with
/// `verified`: each write that fails makes the verdict `Inconclusive` with its error, and the
/// manifest and the log record the final one. A report already written stays (05 §6).
pub(crate) fn finish(
    mut verdict: Verdict,
    execution: Execution,
    out: Option<&str>,
    flags: Flags,
    checked_output: Result<(), ErrorCode>,
    mut trail: Trail,
    manifest: Option<Manifest<'_>>,
) -> CommandOutput {
    trail.flush();
    let mut report = match json(&verdict, &execution) {
        Ok(report) => report,
        Err(code) => return failure(code),
    };
    let mut written = None;
    if let (Ok(()), Some(out)) = (checked_output, out) {
        match files::write(Path::new(out), report.as_bytes()) {
            Ok(()) => written = Some(out),
            Err(code) => {
                verdict = inconclusive(code);
                report = match json(&verdict, &execution) {
                    Ok(report) => report,
                    Err(code) => return failure(code),
                };
            }
        }
    }
    if let (Ok(()), Some(m)) = (checked_output, &manifest) {
        let text = m.render(&verdict, written, (&execution, &trail));
        if let Err(code) = files::write(Path::new(m.path), text.as_bytes()) {
            verdict = inconclusive(code);
            report = match json(&verdict, &execution) {
                Ok(report) => report,
                Err(code) => return failure(code),
            };
        }
    }
    let exit_code = verdict.exit_code();
    trail.verified(&verdict, exit_code, execution.start.elapsed());
    let stderr = trail.into_lines();
    if flags.json {
        return CommandOutput {
            exit_code,
            stdout: report,
            stderr,
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
        stderr,
    }
}
