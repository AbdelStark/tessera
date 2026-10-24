// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `tessera-verify` command-line interface and its JSON verdict report (RFC-0019, 02 §2.3).
//!
//! NI verification authenticates a caller-owned statement pin before running the independent
//! core. `params` retains the independent soundness calculator (RFC-0014).

#![forbid(unsafe_code)]

mod directory;
mod files;
mod live;
mod manifest;
mod observability;
mod options;
mod tables;
mod verify;

use std::fmt::Write;
use tv_core::soundness::{
    DEFAULT_EXP_LOOKUPS, DEFAULT_EXP_ROWS, LookupSize, SoundnessParams, SoundnessReport,
    StatementSizes,
};

/// Exit code on success (`02 §2.4`).
pub const EXIT_SUCCESS: i32 = 0;

/// Exit code on a rejected proof or missed soundness target (`02 §2.4`, `RFC-0051`).
pub const EXIT_REJECT: i32 = 10;

/// Exit code on an operational/resource failure (`02 §2.4`).
pub const EXIT_FAILURE: i32 = 20;

/// Exit code on a usage error (`02 §2.4`, `ERR-USAGE`).
pub const EXIT_USAGE: i32 = 64;

/// Output produced by a `tessera-verify` command invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandOutput {
    /// Process exit status (`02 §2.4`).
    pub exit_code: i32,
    /// Text written to standard output.
    pub stdout: String,
    /// Text written to standard error.
    pub stderr: String,
}

const ROOT_HELP: &str = "\
tessera-verify: independent verifier for the Tessera proof system (RFC-0019)

USAGE:
    tessera-verify <COMMAND> [OPTIONS]

COMMANDS:
    verify    Verify one protocol/1 NI proof against caller-pinned inputs
    tables    Regenerate the claim list's table catalog and print digests
    params    Compute the independent soundness table for a parameter set (RFC-0014)
    live      Verify one caller-pinned Live session through a directory
    weights link, audit plan/report    Reserved; require later ledger/anchor work
";

const PARAMS_HELP: &str = "\
tessera-verify params: independent soundness calculator (RFC-0014 'Soundness calculator')

Evaluates a parameter set against the default Llama-3.1-405B 16K-token, 16K-context statement
(Exp table default: 2^42 lookups and 2^22 rows) and prints every row of the soundness table.

USAGE:
    tessera-verify params --params <PARAMSET> [--lookups <N>] [--table-rows <T>] [--json]

OPTIONS:
    --params <PARAMSET>     Named parameter set: live-64, ni-100, or test-small
    --lookups <N>           Override LogUp lookups N (default when --table-rows is set: 2^42)
    --table-rows <T>        Override LogUp table rows T (default when --lookups is set: 2^22)
    --json                  Print one tessera.soundness/1 JSON object to stdout
    -h, --help              Print this help information
";

/// Runs the `tessera-verify` CLI on `args` (excluding `argv[0]`) and returns its exit code,
/// `stdout`, and `stderr`.
pub fn run<I, S>(args: I) -> CommandOutput
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_inner(args)))
        .unwrap_or_else(|_| verify::failure(tv_core::verdict::ErrorCode::Internal))
}
fn run_inner<I, S>(args: I) -> CommandOutput
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut collected = Vec::new();
    let mut json_flag = false;
    let mut oversized = false;
    for (i, arg) in args.into_iter().take(65).enumerate() {
        let value = arg.as_ref();
        json_flag |= value == "--json";
        if i == 64 || value.len() > 8192 {
            oversized = true;
        } else {
            collected.push(String::from(value));
        }
    }
    if oversized {
        let mut output = usage_error(
            "tessera-verify",
            "too many or oversized arguments",
            json_flag,
        );
        if collected.first().is_some_and(|a| a == "verify")
            && collected
                .windows(2)
                .any(|a| a[0] == "--log-format" && a[1] == "json")
        {
            output.stderr = observability::Observation::new().logs(output.exit_code);
        }
        return output;
    }

    let Some(first) = collected.first().map(String::as_str) else {
        return usage_error("tessera-verify", "missing command", json_flag);
    };

    match first {
        "-h" | "--help" | "help" => CommandOutput {
            exit_code: EXIT_SUCCESS,
            stdout: String::from(ROOT_HELP),
            stderr: String::new(),
        },
        "params" => run_params(&collected[1..], json_flag),
        "verify" => verify::run(&collected[1..], json_flag),
        "tables" => tables::run(&collected[1..], json_flag),
        "live" => live::run(&collected[1..], json_flag),
        "weights" | "audit" => usage_error(
            first,
            "this operation requires unfinished live transport, anchor or ledger support",
            json_flag,
        ),
        other => usage_error(
            "tessera-verify",
            &format!("unknown command `{other}`"),
            json_flag,
        ),
    }
}

fn run_params(args: &[String], json_flag: bool) -> CommandOutput {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        return CommandOutput {
            exit_code: EXIT_SUCCESS,
            stdout: String::from(PARAMS_HELP),
            stderr: String::new(),
        };
    }

    let mut param_name: Option<&str> = None;
    let mut lookups: Option<u64> = None;
    let mut table_rows: Option<u64> = None;

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "--json" => {
                i += 1;
            }
            "--params" => {
                let Some(val) = args.get(i + 1) else {
                    return usage_error("params", "missing value for `--params`", json_flag);
                };
                if param_name.is_some() {
                    return usage_error("params", "`--params` specified more than once", json_flag);
                }
                param_name = Some(val.as_str());
                i += 2;
            }
            "--lookups" => {
                let Some(val) = args.get(i + 1) else {
                    return usage_error("params", "missing value for `--lookups`", json_flag);
                };
                if lookups.is_some() {
                    return usage_error(
                        "params",
                        "`--lookups` specified more than once",
                        json_flag,
                    );
                }
                let Ok(n) = val.parse::<u64>() else {
                    return usage_error(
                        "params",
                        &format!("invalid `--lookups` value `{val}`"),
                        json_flag,
                    );
                };
                if n == 0 {
                    return usage_error("params", "`--lookups` must be at least 1", json_flag);
                }
                lookups = Some(n);
                i += 2;
            }
            "--table-rows" => {
                let Some(val) = args.get(i + 1) else {
                    return usage_error("params", "missing value for `--table-rows`", json_flag);
                };
                if table_rows.is_some() {
                    return usage_error(
                        "params",
                        "`--table-rows` specified more than once",
                        json_flag,
                    );
                }
                let Ok(t) = val.parse::<u64>() else {
                    return usage_error(
                        "params",
                        &format!("invalid `--table-rows` value `{val}`"),
                        json_flag,
                    );
                };
                if t == 0 {
                    return usage_error("params", "`--table-rows` must be at least 1", json_flag);
                }
                table_rows = Some(t);
                i += 2;
            }
            other => {
                return usage_error(
                    "params",
                    &format!("unexpected argument `{other}`"),
                    json_flag,
                );
            }
        }
    }

    let Some(name) = param_name else {
        return usage_error(
            "params",
            "missing required `--params <PARAMSET>`",
            json_flag,
        );
    };
    let Some(params) = SoundnessParams::by_cli_name(name) else {
        return usage_error(
            "params",
            &format!("unknown parameter set `{name}` (expected live-64, ni-100, or test-small)"),
            json_flag,
        );
    };

    let mut statement = StatementSizes::default_405b();
    if lookups.is_some() || table_rows.is_some() {
        statement.lookups = vec![LookupSize {
            lookups: lookups.unwrap_or(DEFAULT_EXP_LOOKUPS),
            rows: table_rows.unwrap_or(DEFAULT_EXP_ROWS),
        }];
    }

    match SoundnessReport::evaluate(params, statement) {
        Ok(report) => {
            let exit_code = match report.meets_targets {
                Some(false) => EXIT_REJECT,
                Some(true) | None => EXIT_SUCCESS,
            };
            let stdout = if json_flag {
                report.to_json()
            } else {
                report.to_table()
            };
            CommandOutput {
                exit_code,
                stdout,
                stderr: String::new(),
            }
        }
        Err(err) => command_error(
            "params",
            &err.to_string(),
            json_flag,
            err.code(),
            if err == tv_core::soundness::SoundnessError::Memory {
                EXIT_FAILURE
            } else {
                EXIT_USAGE
            },
        ),
    }
}

fn usage_error(command: &str, message: &str, json_flag: bool) -> CommandOutput {
    command_error(command, message, json_flag, "ERR-USAGE", EXIT_USAGE)
}
fn command_error(
    command: &str,
    message: &str,
    json_flag: bool,
    code: &str,
    exit_code: i32,
) -> CommandOutput {
    if json_flag {
        let escaped = escape_json(message);
        let stdout = format!(
            "{{\"command\":\"{command}\",\"error\":{{\"code\":\"{code}\",\"message\":\"{escaped}\"}}}}\n"
        );
        CommandOutput {
            exit_code,
            stdout,
            stderr: String::new(),
        }
    } else {
        CommandOutput {
            exit_code,
            stdout: String::new(),
            stderr: format!("error ({code}): {message}\n"),
        }
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{00}'..='\u{1f}' => {
                let _ = write!(out, "\\u{:04x}", u32::from(ch));
            }
            c => out.push(c),
        }
    }
    out
}
