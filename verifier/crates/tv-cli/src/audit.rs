// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera-verify audit plan` and `audit report` (02 §2.3, RFC-0016 'Calculators' and 'Audit
//! report').

use std::path::Path;

use tv_codec::DecodeLimits;
use tv_core::anchor::Failure;
use tv_core::audit::{AuditPolicy, AuditReport, AuditSample, RateBucket, VerifierLedger};
use tv_core::audit_plan::{AuditPlan, CostModel, TrafficBucket};
use tv_core::claims::ParamSetId;
use tv_core::verdict::ErrorCode;

use crate::json::Value;
use crate::options::Options;
use crate::{CommandOutput, EXIT_REJECT, EXIT_SUCCESS, command_error, files};

const HELP: &str = "\
tessera-verify audit: the audit's calculators and accounting (RFC-0016)

USAGE:
  tessera-verify audit plan --policy POLICY.json --traffic PROFILE.json [--mu MU.json] [--json]
  tessera-verify audit report --ledger LEDGER.tvl --sample SAMPLE.tas --out REPORT.json [--json]

plan    Detection probability of a tessera.audit-policy/1 policy at a tessera.audit-traffic/1
        profile, against a prover that pads violations into a later bucket and in place. With
        --mu, a tessera.cost-model/1 report at the traffic's mean lengths, also the amortized
        overhead and each bucket's rate for the report's target overhead. --json prints
        tessera.audit-plan/1; without --mu its overhead figures are null.
report  The tessera.audit-report/1 report of a sample at its deadline, from the verifier's
        ledger: each selected request the ledger records accepted is accepted, with its U-hat,
        and every other one unproven (DeadlineMissed). The sample must be the verifier's own
        derivation. Exit 0 when every selected request is accepted, 10 otherwise.
";

/// The most bytes a JSON input of `audit plan` may hold: a policy, a profile and a cost model
/// take kilobytes, and the parsed tree takes a small multiple of its text.
const JSON_LIMIT: u64 = 1 << 20;
/// The most bytes the sample and the ledger may hold together.
const ARTIFACT_LIMIT: u64 = 1 << 30;

pub(crate) fn run(args: &[String], json_flag: bool) -> CommandOutput {
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
        Some(other) => command_error(
            "audit",
            &format!("unknown audit command `{other}`"),
            json_flag,
            ErrorCode::Usage.id(),
            ErrorCode::Usage.exit_code(),
        ),
    }
}

fn fail(command: &str, message: &str, json_flag: bool, code: ErrorCode) -> CommandOutput {
    command_error(command, message, json_flag, code.id(), code.exit_code())
}

fn read_json(path: &str, input: &mut files::Inputs) -> Result<Value, ErrorCode> {
    let bytes = input.read(Path::new(path))?;
    let text = core::str::from_utf8(&bytes).map_err(|_| ErrorCode::Usage)?;
    Value::parse(text).ok_or(ErrorCode::Usage)
}

// ----- audit plan ------------------------------------------------------------------------------

fn plan(args: &[String], json_flag: bool) -> CommandOutput {
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
    let stdout = if json_flag {
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

fn report(args: &[String], json_flag: bool) -> CommandOutput {
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
    let ledger = match input.read(Path::new(ledger)).and_then(|bytes| {
        VerifierLedger::open(&bytes, &limits).map_err(|e| ErrorCode::from(e.code()))
    }) {
        Ok(ledger) => ledger,
        Err(code) => return fail(COMMAND, "cannot open the ledger", json_flag, code),
    };
    let report = match AuditReport::from_ledger(&sample, &ledger) {
        Ok(report) => report,
        Err(Failure::Inconclusive(e)) => {
            let message = match e.code {
                ErrorCode::ProtocolCoinToss => "the sample's coin toss does not open",
                ErrorCode::Usage => "the ledger's record does not fit the sample",
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
    let stdout = if json_flag {
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
