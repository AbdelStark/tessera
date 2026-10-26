// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera-verify audit plan`, `audit report`, `audit commit` and `audit reveal` (02 §2.3,
//! RFC-0016 'Calculators', 'Sample derivation' and 'Audit report', RFC-0067 'The pending toss').

use std::path::Path;

use tv_codec::DecodeLimits;
use tv_core::anchor::{CheckedManifest, Failure, LogManifest, RecorderKeys, VerifiedLog, Window};
use tv_core::audit::{
    AuditPolicy, AuditReport, AuditSample, LedgerFile, RateBucket, TossFile, read_ledger,
};
use tv_core::audit_plan::{AuditPlan, CostModel, TrafficBucket};
use tv_core::claims::ParamSetId;
use tv_core::verdict::{ErrorCode, VerifierError};

use crate::json::Value;
use crate::log::Flags;
use crate::options::{self, Options};
use crate::{CommandOutput, EXIT_REJECT, EXIT_SUCCESS, command_error, files};

const HELP: &str = "\
tessera-verify audit: the audit's calculators, coin toss and accounting (RFC-0016, RFC-0067)

USAGE:
  tessera-verify audit plan --policy POLICY.json --traffic PROFILE.json [--mu MU.json] [--json]
  tessera-verify audit report --ledger LEDGER.tvl --sample SAMPLE.tas --out REPORT.json [--json]
  tessera-verify audit commit --manifest MANIFEST.tlm --recorder-key KEY --policy POLICY.json
                              --toss TOSS.tvt [--json]
  tessera-verify audit reveal --toss TOSS.tvt --prover-share HEX --manifest MANIFEST.tlm
                              --recorder-key KEY --log LOG.tdl
                              [--previous-manifest M.tlm --previous-log L.tdl]
                              [--between MANIFEST.tlm]... --ledger LEDGER.tvl [--new-ledger]
                              --out SAMPLE.tas [--json]

commit  The verifier's commitment c_V for the coin toss over a manifest whose signature passes
        under the pinned recorder key KEY, 64 hexadecimal digits of its Ed25519 public key. A
        fresh share and the tessera.audit-policy/1 policy are kept in TOSS.tvt, created with
        mode 0600 and synced, before c_V is printed. An existing TOSS.tvt exits 64 and is never
        replaced. The share is secret until the reveal and is never printed.
reveal  The period's sample. The period's log is checked against its manifest, and the
        previous period's when it is named; each --between manifest, a period of zero length
        between the two, in chain order, is checked under the pinned key, and the chain must
        bind them (else exit 64). The toss, read from TOSS.tvt, must be over the period's
        manifest (else exit 64) and must open its commitment (a broken envelope exits 65, a
        commitment the share does not open 20). It is revealed against the prover's share HEX,
        the sample is recorded in the ledger, then written to SAMPLE.tas, and TOSS.tvt is
        overwritten with zeros and removed. The ledger must exist (a missing one exits 20),
        unless --new-ledger creates it, which an existing file at its path refuses (exit 64).
        A reveal interrupted after the record is run again with the same prover share, without
        --new-ledger.

plan    Detection probability of a tessera.audit-policy/1 policy at a tessera.audit-traffic/1
        profile, against a prover that pads violations into a later bucket and in place. With
        --mu, a tessera.cost-model/1 report at the traffic's mean lengths, also the amortized
        overhead and each bucket's rate for the report's target overhead. --json prints
        tessera.audit-plan/1; without --mu its overhead figures are null.
report  The tessera.audit-report/1 report of a sample at its deadline, from the verifier's
        ledger: each selected request the ledger records accepted is accepted, with its U-hat,
        and every other one unproven (DeadlineMissed). The sample must be one the ledger records
        as the verifier's own derivation, else exit 64. Exit 0 when every selected request is
        accepted, 10 otherwise.
";

/// The most bytes a JSON input of `audit plan` may hold: a policy, a profile and a cost model
/// take kilobytes, and the parsed tree takes a small multiple of its text.
const JSON_LIMIT: u64 = 1 << 20;
/// The most bytes the sample, or a period's log, may hold. The ledger is read under
/// `tv_core::audit::LEDGER_LIMITS`, as its writers keep it, and the toss under `TOSS_LIMITS`.
const ARTIFACT_LIMIT: u64 = 1 << 30;
/// The most bytes a manifest may hold: an honest one is 310, and its signature's fields are
/// checked for their lengths once it decodes.
const MANIFEST_LIMIT: u64 = 1 << 16;

/// `json_flag` carries `--json` with the log options, which refusals are written in (05 §2).
pub(crate) fn run(args: &[String], json_flag: Flags) -> CommandOutput {
    match args.first().map(String::as_str) {
        None | Some("-h" | "--help" | "help") => CommandOutput {
            exit_code: EXIT_SUCCESS,
            stdout: HELP.into(),
            stderr: String::new(),
        },
        Some(_) if matches!(args.get(1).map(String::as_str), Some("-h" | "--help")) => {
            CommandOutput {
                exit_code: EXIT_SUCCESS,
                stdout: HELP.into(),
                stderr: String::new(),
            }
        }
        Some("plan") => plan(&args[1..], json_flag),
        Some("report") => report(&args[1..], json_flag),
        Some("commit") => commit(&args[1..], json_flag).unwrap_or_else(|refused| refused),
        Some("reveal") => reveal(&args[1..], json_flag).unwrap_or_else(|refused| refused),
        Some(other) => command_error(
            "audit",
            &format!("unknown audit command `{other}`"),
            json_flag,
            ErrorCode::Usage.id(),
            ErrorCode::Usage.exit_code(),
        ),
    }
}

fn fail(command: &str, message: &str, json_flag: Flags, code: ErrorCode) -> CommandOutput {
    command_error(command, message, json_flag, code.id(), code.exit_code())
}

fn read_json(path: &str, input: &mut files::Inputs) -> Result<Value, ErrorCode> {
    let bytes = input.read(Path::new(path))?;
    let text = core::str::from_utf8(&bytes).map_err(|_| ErrorCode::Usage)?;
    Value::parse(text).ok_or(ErrorCode::Usage)
}

// ----- audit plan ------------------------------------------------------------------------------

fn plan(args: &[String], json_flag: Flags) -> CommandOutput {
    const COMMAND: &str = "audit plan";
    let options = match Options::parse(args, &["--policy", "--traffic", "--mu"]) {
        Ok(o) => o,
        Err(code) => {
            return fail(
                COMMAND,
                "invalid, missing or duplicate option",
                json_flag,
                code,
            );
        }
    };
    let (Ok(policy_path), Ok(traffic_path)) =
        (options.required("--policy"), options.required("--traffic"))
    else {
        return fail(
            COMMAND,
            "missing --policy or --traffic",
            json_flag,
            ErrorCode::Usage,
        );
    };
    let mut input = files::Inputs {
        remaining: 3 * JSON_LIMIT,
        max_artifact: JSON_LIMIT,
    };
    let policy = match read_json(policy_path, &mut input).map(|v| read_policy(&v)) {
        Ok(Ok(policy)) => policy,
        Ok(Err(message)) => return fail(COMMAND, &message, json_flag, ErrorCode::Usage),
        Err(code) => return fail(COMMAND, "cannot read the policy", json_flag, code),
    };
    let traffic = match read_json(traffic_path, &mut input).map(|v| read_traffic(&v)) {
        Ok(Some(traffic)) => traffic,
        Ok(None) => {
            return fail(
                COMMAND,
                "the traffic profile is not tessera.audit-traffic/1",
                json_flag,
                ErrorCode::Usage,
            );
        }
        Err(code) => return fail(COMMAND, "cannot read the traffic profile", json_flag, code),
    };
    let cost = match options.get("--mu") {
        None => None,
        Some(path) => match read_json(path, &mut input).map(|v| read_cost_model(&v)) {
            Ok(Some(cost)) => Some(cost),
            Ok(None) => {
                return fail(
                    COMMAND,
                    "--mu is not a tessera.cost-model/1 report",
                    json_flag,
                    ErrorCode::Usage,
                );
            }
            Err(code) => return fail(COMMAND, "cannot read the cost model", json_flag, code),
        },
    };
    let plan = match AuditPlan::new(&policy, &traffic, cost.as_ref()) {
        Ok(plan) => plan,
        Err(e) => return fail(COMMAND, &e.to_string(), json_flag, e.code()),
    };
    let stdout = if json_flag.json {
        match plan.to_json() {
            Ok(text) => text,
            Err(e) => return fail(COMMAND, "cannot render the plan", json_flag, e.code),
        }
    } else {
        plan.to_table()
    };
    CommandOutput {
        exit_code: EXIT_SUCCESS,
        stdout,
        stderr: String::new(),
    }
}

/// A `tessera.audit-policy/1` file, read strictly: every key, no other (RFC-0016 'Policy
/// files'). Each refusal is a message for `ERR-USAGE`.
fn read_policy(v: &Value) -> Result<AuditPolicy, String> {
    let refuse = |what: &str| format!("the policy {what}");
    let [schema, period, buckets, min, max, deadline, params] = v
        .exactly([
            "schema",
            "period_ns",
            "buckets",
            "min_batch_tokens",
            "max_batch_tokens",
            "proof_deadline_ns",
            "params",
        ])
        .ok_or_else(|| refuse("does not have exactly the keys of tessera.audit-policy/1"))?;
    if schema.as_str() != Some("tessera.audit-policy/1") {
        return Err(refuse("is not tessera.audit-policy/1"));
    }
    let u32_of = |v: &Value| v.as_u64().and_then(|n| u32::try_from(n).ok());
    let mut rates = Vec::new();
    for (i, bucket) in buckets
        .as_array()
        .ok_or_else(|| refuse("has no bucket array"))?
        .iter()
        .enumerate()
    {
        let [max_tokens, rate] = bucket
            .exactly(["max_tokens", "rate"])
            .ok_or_else(|| refuse(&format!("bucket {i} is not {{max_tokens, rate}}")))?;
        let max_tokens =
            u32_of(max_tokens).ok_or_else(|| refuse(&format!("bucket {i}'s max_tokens")))?;
        let threshold = rate
            .as_str()
            .and_then(RateBucket::threshold_of)
            .ok_or_else(|| refuse(&format!("bucket {i}'s rate is not a decimal rate")))?;
        rates.push(RateBucket {
            max_tokens,
            threshold,
        });
    }
    let params = params
        .as_str()
        .filter(|s| !s.is_empty())
        .and_then(ParamSetId::new)
        .ok_or_else(|| refuse("params is not printable ASCII"))?;
    AuditPolicy::new(
        period.as_u64().ok_or_else(|| refuse("period_ns"))?,
        rates,
        u32_of(min).ok_or_else(|| refuse("min_batch_tokens"))?,
        u32_of(max).ok_or_else(|| refuse("max_batch_tokens"))?,
        deadline
            .as_u64()
            .ok_or_else(|| refuse("proof_deadline_ns"))?,
        params,
    )
    .map_err(|e| refuse(&format!("breaks a rule: {e}")))
}

/// A `tessera.audit-traffic/1` profile, read strictly.
fn read_traffic(v: &Value) -> Option<Vec<TrafficBucket>> {
    let [schema, buckets] = v.exactly(["schema", "buckets"])?;
    if schema.as_str() != Some("tessera.audit-traffic/1") {
        return None;
    }
    buckets
        .as_array()?
        .iter()
        .map(|b| {
            let [requests, tokens, violations] = b.exactly(["requests", "tokens", "violations"])?;
            Some(TrafficBucket {
                requests: requests.as_u64()?,
                tokens: tokens.as_u64().and_then(|n| u32::try_from(n).ok())?,
                violations: violations.as_u64()?,
            })
        })
        .collect()
}

/// What the plan needs of a `tessera.cost-model/1` report: `inputs.source_defaults`, the
/// scenario's `eta` and `target_overhead`, and each projection's `context` and `mu.total`. The
/// rest is the prover's and is not read.
fn read_cost_model(v: &Value) -> Option<CostModel> {
    if v.get("schema")?.as_str()? != "tessera.cost-model/1" {
        return None;
    }
    let inputs = v.get("inputs")?;
    let scenario = inputs.get("scenario")?;
    let mu = v
        .get("projections")?
        .as_array()?
        .iter()
        .map(|p| {
            Some((
                p.get("context")?.as_u64()?,
                p.get("mu")?.get("total")?.as_f64()?,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(CostModel {
        source_defaults: inputs.get("source_defaults")?.as_bool()?,
        eta: scenario.get("eta")?.as_f64()?,
        target_overhead: scenario.get("target_overhead")?.as_f64()?,
        mu,
    })
}

// ----- audit report ----------------------------------------------------------------------------

fn report(args: &[String], json_flag: Flags) -> CommandOutput {
    const COMMAND: &str = "audit report";
    let options = match Options::parse(args, &["--ledger", "--sample", "--out"]) {
        Ok(o) => o,
        Err(code) => {
            return fail(
                COMMAND,
                "invalid, missing or duplicate option",
                json_flag,
                code,
            );
        }
    };
    let (Ok(ledger), Ok(sample), Ok(out)) = (
        options.required("--ledger"),
        options.required("--sample"),
        options.required("--out"),
    ) else {
        return fail(
            COMMAND,
            "missing --ledger, --sample or --out",
            json_flag,
            ErrorCode::Usage,
        );
    };
    if let Err(code) = files::output_distinct(Some(out), &[ledger, sample]) {
        return fail(COMMAND, "--out names an input", json_flag, code);
    }
    let mut input = files::Inputs {
        remaining: ARTIFACT_LIMIT,
        max_artifact: ARTIFACT_LIMIT,
    };
    let limits = DecodeLimits {
        max_total_bytes: ARTIFACT_LIMIT,
        ..DecodeLimits::MESSAGE
    };
    let sample = match input
        .read(Path::new(sample))
        .and_then(|bytes| AuditSample::open(&bytes, &limits).map_err(|e| ErrorCode::from(e.code())))
    {
        Ok(sample) => sample,
        Err(code) => return fail(COMMAND, "cannot open the sample", json_flag, code),
    };
    // The ledger's own reader: a regular file of one name, refused before it is opened
    // otherwise (RFC-0067 'The path'), under the limits every writer keeps it within (RFC-0067
    // 'One limit').
    let ledger = match read_ledger(Path::new(ledger)) {
        Ok(ledger) => ledger,
        Err(e) => return fail(COMMAND, "cannot open the ledger", json_flag, e.code),
    };
    let report = match AuditReport::from_ledger(&sample, &ledger) {
        Ok(report) => report,
        Err(Failure::Inconclusive(e)) => {
            let message = match e.code {
                ErrorCode::ProtocolCoinToss => "the sample's coin toss does not open",
                ErrorCode::Usage => {
                    "the sample is not one this ledger derived, or its record does not fit it"
                }
                _ => "cannot build the report",
            };
            return fail(COMMAND, message, json_flag, e.code);
        }
        Err(Failure::Rejected(_)) => {
            return fail(
                COMMAND,
                "cannot build the report",
                json_flag,
                ErrorCode::Internal,
            );
        }
    };
    let text = match report.to_json() {
        Ok(text) => text,
        Err(e) => return fail(COMMAND, "cannot render the report", json_flag, e.code),
    };
    if let Err(code) = files::write(Path::new(out), text.as_bytes()) {
        return fail(COMMAND, "cannot write the report", json_flag, code);
    }
    let s = report.summary();
    let exit_code = if s.accepted == s.selected {
        EXIT_SUCCESS
    } else {
        EXIT_REJECT
    };
    let stdout = if json_flag.json {
        text
    } else {
        format!(
            "selected {}, accepted {}, rejected {}, unproven {}, inconclusive {}\n",
            s.selected, s.accepted, s.rejected, s.unproven, s.inconclusive
        )
    };
    CommandOutput {
        exit_code,
        stdout,
        stderr: String::new(),
    }
}

// ----- audit commit and audit reveal -----------------------------------------------------------

/// A refusal, already rendered as the command's output.
type Refused = CommandOutput;

/// `bytes` as lowercase hexadecimal.
fn hex_of(bytes: &[u8]) -> String {
    use core::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// A failure of the core as the command's output: an error exits with its code, a rejection
/// with 10.
fn failure(command: &str, message: &str, json_flag: Flags, failure: Failure) -> Refused {
    match failure {
        Failure::Inconclusive(e) => fail(command, message, json_flag, e.code),
        Failure::Rejected(reason) => {
            command_error(command, message, json_flag, reason.code(), EXIT_REJECT)
        }
    }
}

/// Reads `path`, at most `limit` bytes.
fn read_input(path: &str, limit: u64) -> Result<Vec<u8>, ErrorCode> {
    files::Inputs {
        remaining: limit,
        max_artifact: limit,
    }
    .read(Path::new(path))
}

/// The recorder key the verifier pins (`--recorder-key`, RFC-0015 'Signatures'): the 32 bytes of
/// its Ed25519 public key, as 64 hexadecimal digits.
fn recorder_keys(command: &str, key: &str, json_flag: Flags) -> Result<RecorderKeys, Refused> {
    options::hex::<32>(key)
        .map(|key| RecorderKeys::new(vec![key]))
        .map_err(|code| {
            fail(
                command,
                "--recorder-key is not 64 hexadecimal digits",
                json_flag,
                code,
            )
        })
}

/// The manifest at `path`, decoded and checked under `keys`, its signature first (RFC-0060
/// 'Signatures'). A manifest that fails its own checks is the anchor's rejection, exit 10.
fn checked_manifest(
    command: &str,
    path: &str,
    keys: &RecorderKeys,
    json_flag: Flags,
    what: &str,
) -> Result<CheckedManifest, Refused> {
    let bytes = read_input(path, MANIFEST_LIMIT)
        .map_err(|code| fail(command, &format!("cannot read {what}"), json_flag, code))?;
    let limits = DecodeLimits {
        max_total_bytes: MANIFEST_LIMIT,
        ..DecodeLimits::MESSAGE
    };
    let sealed = LogManifest::open(&bytes, &limits).map_err(|e| {
        fail(
            command,
            &format!("{what} does not decode"),
            json_flag,
            e.code().into(),
        )
    })?;
    sealed.check(keys).map_err(|e| {
        command_error(
            command,
            &format!("{what} is refused: {e}"),
            json_flag,
            e.reason().code(),
            EXIT_REJECT,
        )
    })
}

/// The period's log at `path`, checked against `manifest` (RFC-0062 'The check'). A log that is
/// not its manifest's exits 65, `ERR-CODEC-DIGEST`.
fn verified_log(
    command: &str,
    manifest: &CheckedManifest,
    path: &str,
    json_flag: Flags,
    what: &str,
) -> Result<VerifiedLog, Refused> {
    let bytes = read_input(path, ARTIFACT_LIMIT)
        .map_err(|code| fail(command, &format!("cannot read {what}"), json_flag, code))?;
    let limits = DecodeLimits {
        max_total_bytes: ARTIFACT_LIMIT,
        ..DecodeLimits::MESSAGE
    };
    VerifiedLog::check(manifest, &bytes, &limits).map_err(|e| {
        fail(
            command,
            &format!("{what} is not its manifest's"),
            json_flag,
            e.code().into(),
        )
    })
}

/// `audit commit` (02 §2.3, RFC-0067 'Before the commitment leaves'): the manifest's signature,
/// the policy, then the toss written to its file, and only then the commitment printed.
fn commit(args: &[String], json_flag: Flags) -> Result<CommandOutput, Refused> {
    const COMMAND: &str = "audit commit";
    let options = Options::parse(
        args,
        &["--manifest", "--recorder-key", "--policy", "--toss"],
    )
    .map_err(|code| {
        fail(
            COMMAND,
            "invalid, missing or duplicate option",
            json_flag,
            code,
        )
    })?;
    let (Ok(manifest), Ok(key), Ok(policy), Ok(toss)) = (
        options.required("--manifest"),
        options.required("--recorder-key"),
        options.required("--policy"),
        options.required("--toss"),
    ) else {
        return Err(fail(
            COMMAND,
            "missing --manifest, --recorder-key, --policy or --toss",
            json_flag,
            ErrorCode::Usage,
        ));
    };
    let keys = recorder_keys(COMMAND, key, json_flag)?;
    let manifest = checked_manifest(COMMAND, manifest, &keys, json_flag, "the manifest")?;
    let mut input = files::Inputs {
        remaining: JSON_LIMIT,
        max_artifact: JSON_LIMIT,
    };
    let policy = match read_json(policy, &mut input).map(|v| read_policy(&v)) {
        Ok(Ok(policy)) => policy,
        Ok(Err(message)) => return Err(fail(COMMAND, &message, json_flag, ErrorCode::Usage)),
        Err(code) => return Err(fail(COMMAND, "cannot read the policy", json_flag, code)),
    };
    // The toss is in its file, synced, before the commitment exists outside this process.
    let pending = TossFile::commit(Path::new(toss), &manifest, policy).map_err(|e| {
        let message = match e.code {
            ErrorCode::Usage => {
                "--toss names an existing file or no file: a pending toss is never replaced"
            }
            ErrorCode::CodecLimit => "the policy makes the toss larger than its limit",
            ErrorCode::Io => "cannot draw the share, or write and sync the toss",
            _ => "cannot keep the toss",
        };
        fail(COMMAND, message, json_flag, e.code)
    })?;
    let commitment = hex_of(pending.commitment());
    let stdout = if json_flag.json {
        format!(
            "{{\"command\":\"{COMMAND}\",\"manifest_digest\":\"{}\",\"commitment\":\"{commitment}\"}}\n",
            hex_of(pending.manifest_digest())
        )
    } else {
        format!("{commitment}\n")
    };
    Ok(CommandOutput {
        exit_code: EXIT_SUCCESS,
        stdout,
        stderr: String::new(),
    })
}

/// The ledger at `path`, held by this writer. With `new`, it is created, exclusively: anything
/// at the path, a dangling link included, is `ERR-USAGE` and is left as it is. Without it, the
/// ledger must exist, and a missing one is `ERR-RES-IO` (RFC-0067 'The path'), so that a
/// mistyped path never makes a second ledger.
fn open_ledger(path: &Path, new: bool) -> Result<LedgerFile, VerifierError> {
    if new {
        LedgerFile::create(path)
    } else {
        LedgerFile::open(path)
    }
}

/// `audit reveal` (02 §2.3, RFC-0016 'Sample derivation', RFC-0063 'The previous period's log',
/// RFC-0067 'The pending toss'): both periods' logs checked and bound through the zero-length
/// periods between them, the toss read and revealed, the sample recorded in the ledger, then
/// written, then the toss cleared.
fn reveal(args: &[String], json_flag: Flags) -> Result<CommandOutput, Refused> {
    const COMMAND: &str = "audit reveal";
    let options = Options::parse_with(
        args,
        &[
            "--toss",
            "--prover-share",
            "--manifest",
            "--recorder-key",
            "--log",
            "--previous-manifest",
            "--previous-log",
            "--ledger",
            "--out",
        ],
        &["--between"],
        &["--new-ledger"],
    )
    .map_err(|code| {
        fail(
            COMMAND,
            "invalid, missing or duplicate option",
            json_flag,
            code,
        )
    })?;
    let (Ok(toss), Ok(share), Ok(manifest), Ok(key), Ok(log), Ok(ledger), Ok(out)) = (
        options.required("--toss"),
        options.required("--prover-share"),
        options.required("--manifest"),
        options.required("--recorder-key"),
        options.required("--log"),
        options.required("--ledger"),
        options.required("--out"),
    ) else {
        return Err(fail(
            COMMAND,
            "missing --toss, --prover-share, --manifest, --recorder-key, --log, --ledger or --out",
            json_flag,
            ErrorCode::Usage,
        ));
    };
    let previous = match (
        options.get("--previous-manifest"),
        options.get("--previous-log"),
    ) {
        (Some(manifest), Some(log)) => Some((manifest, log)),
        (None, None) => None,
        _ => {
            return Err(fail(
                COMMAND,
                "--previous-manifest and --previous-log are named together",
                json_flag,
                ErrorCode::Usage,
            ));
        }
    };
    let between = options.all("--between");
    let new_ledger = options.switch("--new-ledger");
    let mut inputs = vec![toss, manifest, log, ledger];
    if let Some((manifest, log)) = previous {
        inputs.extend([manifest, log]);
    }
    inputs.extend(&between);
    files::output_distinct(Some(out), &inputs)
        .map_err(|code| fail(COMMAND, "--out names an input", json_flag, code))?;
    let prover_share = options::hex::<32>(share).map_err(|code| {
        fail(
            COMMAND,
            "--prover-share is not 64 hexadecimal digits",
            json_flag,
            code,
        )
    })?;
    let keys = recorder_keys(COMMAND, key, json_flag)?;
    // 'The reveal comes last': the period's manifest and log, and the previous period's, are
    // checked before the toss is read (RFC-0016 'Sample derivation' step 1).
    let manifest = checked_manifest(COMMAND, manifest, &keys, json_flag, "the period's manifest")?;
    let period = verified_log(COMMAND, &manifest, log, json_flag, "the period's log")?;
    let before = match previous {
        Some((manifest, log)) => {
            let manifest = checked_manifest(
                COMMAND,
                manifest,
                &keys,
                json_flag,
                "the previous period's manifest",
            )?;
            Some(verified_log(
                COMMAND,
                &manifest,
                log,
                json_flag,
                "the previous period's log",
            )?)
        }
        None => None,
    };
    // The periods of zero length between the two, in chain order, each signed under the pinned
    // key (RFC-0063 'The previous period's log').
    let between = between
        .iter()
        .map(|path| checked_manifest(COMMAND, path, &keys, json_flag, "a --between manifest"))
        .collect::<Result<Vec<CheckedManifest>, Refused>>()?;
    let window = Window::bind(&period, &between, before.as_ref()).map_err(|f| {
        failure(
            COMMAND,
            "the previous period and the --between manifests do not bind to the period through the chain",
            json_flag,
            f,
        )
    })?;
    let mut pending = TossFile::open(Path::new(toss)).map_err(|e| {
        let message = match e.code {
            ErrorCode::Io => "cannot read the toss; a lost toss cannot be recreated",
            ErrorCode::Usage => "the toss is not a regular file of one name",
            ErrorCode::ProtocolCoinToss => "the toss's share does not open its commitment",
            ErrorCode::CodecDigest => "the toss's envelope digest does not match",
            _ => "the toss does not decode",
        };
        fail(COMMAND, message, json_flag, e.code)
    })?;
    // Another manifest's toss is refused before the ledger is opened or created; the reveal
    // checks it again.
    if pending.manifest_digest() != window.period().manifest_digest() {
        return Err(fail(
            COMMAND,
            "the toss is over another manifest",
            json_flag,
            ErrorCode::Usage,
        ));
    }
    let mut ledger = open_ledger(Path::new(ledger), new_ledger).map_err(|e| {
        let message = match (new_ledger, e.code) {
            (true, ErrorCode::Usage) => {
                "--new-ledger names an existing file: a ledger is never replaced"
            }
            (false, ErrorCode::Io) => {
                "cannot hold the ledger: it is missing, or another writer holds its directory; --new-ledger creates one"
            }
            _ => "cannot hold the ledger as its one writer",
        };
        fail(COMMAND, message, json_flag, e.code)
    })?;
    let created = new_ledger;
    let sample = pending
        .reveal(&window, prover_share, &mut ledger)
        .map_err(|f| {
            let message = match &f {
                Failure::Inconclusive(VerifierError {
                    code: ErrorCode::ProtocolOrder,
                }) => "the ledger holds another sample for this manifest",
                Failure::Inconclusive(VerifierError {
                    code: ErrorCode::Usage,
                }) => "the toss is over another manifest",
                _ => "cannot reveal the toss and record its sample",
            };
            failure(COMMAND, message, json_flag, f)
        })?;
    drop(ledger);
    // The sample leaves only now that the ledger holds it, and the toss is cleared only once
    // the sample is written and its directory synced: a stop before either is followed by the
    // same reveal again, which changes nothing.
    let sealed = sample
        .seal()
        .map_err(|e| fail(COMMAND, "cannot seal the sample", json_flag, e.code))?;
    files::write(Path::new(out), &sealed)
        .and_then(|()| files::sync_parent(Path::new(out)))
        .map_err(|code| {
            fail(
                COMMAND,
                "the ledger records the sample, which cannot be written: reveal again with the same prover share",
                json_flag,
                code,
            )
        })?;
    pending.clear().map_err(|e| {
        fail(
            COMMAND,
            "the sample is written, but the toss cannot be cleared: reveal again with the same prover share",
            json_flag,
            e.code,
        )
    })?;
    let identity = hex_of(sample.identity());
    let selected = sample.selected().len();
    let stdout = if json_flag.json {
        format!(
            "{{\"command\":\"{COMMAND}\",\"manifest_digest\":\"{}\",\"sample\":\"{identity}\",\"selected\":{selected},\"ledger_created\":{created}}}\n",
            hex_of(sample.manifest_digest())
        )
    } else {
        format!("sample {identity}: {selected} requests selected\n")
    };
    Ok(CommandOutput {
        exit_code: EXIT_SUCCESS,
        stdout,
        stderr: String::new(),
    })
}
