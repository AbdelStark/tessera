// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `tessera-verify` command-line interface and its JSON verdict report (RFC-0019, 02 §2.3).
//!
//! NI verification authenticates a caller-owned statement pin before running the independent
//! core. `params` retains the independent soundness calculator (RFC-0014).

#![forbid(unsafe_code)]

mod audit;
mod directory;
mod files;
mod json;
mod live;
mod log;
mod manifest;
mod options;
mod tables;
mod verify;

use log::Flags;

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
    audit     plan: a policy's detection and overhead; report: a sample's audit report;
              commit and reveal: the verifier's coin toss, kept in a file between them
              (RFC-0016, RFC-0067)
    weights link    Reserved; requires the ledger's weight epochs (#158)

OPTIONS OF EVERY COMMAND (05 §2):
    --log-format <human|json>   stderr's lines: human (default) or tessera.log-event/1 objects
    --log-level <LEVEL>         The most detailed level written: error, warn, info (default),
                                debug (one line per check) or trace
";

const PARAMS_HELP: &str = "\
tessera-verify params: independent soundness calculator (RFC-0014 'Soundness calculator')

Evaluates a parameter set against the default Llama-3.1-405B 16K-token, 16K-context statement
(Exp table default: 2^42 lookups and 2^22 rows), with its leaf stage in the most token groups a
verifier accepts for it, 2^14 (RFC-0054), and prints every row of the soundness table.

USAGE:
    tessera-verify params --params <PARAMSET> [--lookups <N>] [--table-rows <T>] [--json]

OPTIONS:
    --params <PARAMSET>     Named parameter set: live-64, ni-100, or test-small
    --lookups <N>           Override LogUp lookups N (default when --table-rows is set: 2^42)
    --table-rows <T>        Override LogUp table rows T (default when --lookups is set: 2^22);
                            past 42 levels, the leaf stage's follow ceil(log2 max(N, T))
    --json                  Print one tessera.soundness/1 JSON object to stdout
    -h, --help              Print this help information
";

/// [`run`] on the process's arguments (excluding `argv[0]`), which may not be UTF-8.
///
/// An argument that is not UTF-8 is refused as before, `ERR-USAGE` with the line ``unknown
/// command `invalid-utf8-argument` ``, but in the format the valid arguments ask for (05 §2):
/// their `--log-format`, `--log-level` and `--json` are kept, and a log option whose own value is
/// not UTF-8 is an invalid value.
pub fn run_os<I>(args: I) -> CommandOutput
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    const PLACEHOLDER: &str = "invalid-utf8-argument";
    let given: Vec<Option<String>> = args.into_iter().map(|a| a.into_string().ok()).collect();
    if given.iter().all(Option::is_some) {
        return run(given.into_iter().flatten());
    }
    let mut kept = Vec::new();
    let mut json = false;
    let mut given = given.into_iter();
    while let Some(arg) = given.next() {
        match arg.as_deref() {
            Some(option @ ("--log-format" | "--log-level")) => {
                kept.push(option.to_string());
                if let Some(value) = given.next() {
                    kept.push(value.unwrap_or_else(|| PLACEHOLDER.to_string()));
                }
            }
            Some("--json") => json = true,
            _ => {}
        }
    }
    // The placeholder stands where the command is read; `--json` counts in any position.
    kept.push(PLACEHOLDER.to_string());
    if json {
        kept.push("--json".to_string());
    }
    run(kept)
}

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
    // An argument past the 64th, or longer than 8 KiB, is not kept: `None` holds its place.
    let mut given = Vec::new();
    let mut json_flag = false;
    let mut oversized = false;
    for (i, arg) in args.into_iter().take(65).enumerate() {
        let value = arg.as_ref();
        json_flag |= value == "--json";
        if i == 64 || value.len() > 8192 {
            oversized = true;
            given.push(None);
        } else {
            given.push(Some(String::from(value)));
        }
    }
    // `--log-format` and `--log-level` are read before anything else, so that every later
    // refusal is logged in the format asked for; their own refusal is human (05 §2).
    let (flags, rest) = match log::scan(given.clone(), json_flag) {
        Ok(scanned) => scanned,
        Err(detail) => return usage_error("tessera-verify", detail, Flags::human(json_flag)),
    };
    if oversized {
        return usage_error("tessera-verify", "too many or oversized arguments", flags);
    }
    let command_line: Vec<String> = given.into_iter().flatten().collect();
    let collected: Vec<String> = rest.into_iter().flatten().collect();

    let Some(first) = collected.first().map(String::as_str) else {
        return usage_error("tessera-verify", "missing command", flags);
    };

    match first {
        "-h" | "--help" | "help" => CommandOutput {
            exit_code: EXIT_SUCCESS,
            stdout: String::from(ROOT_HELP),
            stderr: String::new(),
        },
        "params" => run_params(&collected[1..], flags),
        "verify" => verify::run(&collected[1..], flags, &command_line),
        "tables" => tables::run(&collected[1..], flags),
        "live" => live::run(&collected[1..], flags),
        "audit" => audit::run(&collected[1..], flags),
        "weights" => usage_error(
            first,
            "this operation requires the ledger's weight epochs (#158)",
            flags,
        ),
        other => usage_error(
            "tessera-verify",
            &format!("unknown command `{other}`"),
            flags,
        ),
    }
}

fn run_params(args: &[String], flags: Flags) -> CommandOutput {
    let json_flag = flags.json;
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
                    return usage_error("params", "missing value for `--params`", flags);
                };
                if param_name.is_some() {
                    return usage_error("params", "`--params` specified more than once", flags);
                }
                param_name = Some(val.as_str());
                i += 2;
            }
            "--lookups" => {
                let Some(val) = args.get(i + 1) else {
                    return usage_error("params", "missing value for `--lookups`", flags);
                };
                if lookups.is_some() {
                    return usage_error("params", "`--lookups` specified more than once", flags);
                }
                let Ok(n) = val.parse::<u64>() else {
                    return usage_error(
                        "params",
                        &format!("invalid `--lookups` value `{val}`"),
                        flags,
                    );
                };
                if n == 0 {
                    return usage_error("params", "`--lookups` must be at least 1", flags);
                }
                lookups = Some(n);
                i += 2;
            }
            "--table-rows" => {
                let Some(val) = args.get(i + 1) else {
                    return usage_error("params", "missing value for `--table-rows`", flags);
                };
                if table_rows.is_some() {
                    return usage_error("params", "`--table-rows` specified more than once", flags);
                }
                let Ok(t) = val.parse::<u64>() else {
                    return usage_error(
                        "params",
                        &format!("invalid `--table-rows` value `{val}`"),
                        flags,
                    );
                };
                if t == 0 {
                    return usage_error("params", "`--table-rows` must be at least 1", flags);
                }
                table_rows = Some(t);
                i += 2;
            }
            other => {
                return usage_error("params", &format!("unexpected argument `{other}`"), flags);
            }
        }
    }

    let Some(name) = param_name else {
        return usage_error("params", "missing required `--params <PARAMSET>`", flags);
    };
    let Some(params) = SoundnessParams::by_cli_name(name) else {
        return usage_error(
            "params",
            &format!("unknown parameter set `{name}` (expected live-64, ni-100, or test-small)"),
            flags,
        );
    };

    // RFC-0014 'Soundness calculator': the default statement, at the most groups the verifier
    // accepts for it (RFC-0054 'A statement's report'); `--lookups` and `--table-rows` replace its
    // instances, and the leaf stage's levels then follow the tallest tree.
    let statement = if lookups.is_some() || table_rows.is_some() {
        StatementSizes::default_405b_with_instance(LookupSize {
            lookups: lookups.unwrap_or(DEFAULT_EXP_LOOKUPS),
            rows: table_rows.unwrap_or(DEFAULT_EXP_ROWS),
        })
    } else {
        StatementSizes::default_405b()
    };

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
            flags,
            err.code(),
            if err == tv_core::soundness::SoundnessError::Memory {
                EXIT_FAILURE
            } else {
                EXIT_USAGE
            },
        ),
    }
}

fn usage_error(command: &str, message: &str, flags: Flags) -> CommandOutput {
    command_error(command, message, flags, "ERR-USAGE", EXIT_USAGE)
}
/// A refused command: with `--json`, stdout's `{"command", "error": {"code", "message"}}`
/// (02 §2); on stderr, `tessera.tv_cli.command_failed` in the format asked for, today's
/// `error (CODE): message` in the human one.
pub(crate) fn command_error(
    command: &str,
    message: &str,
    flags: Flags,
    code: &str,
    exit_code: i32,
) -> CommandOutput {
    let stdout = if flags.json {
        let escaped = escape_json(message);
        format!(
            "{{\"command\":\"{command}\",\"error\":{{\"code\":\"{code}\",\"message\":\"{escaped}\"}}}}\n"
        )
    } else {
        String::new()
    };
    CommandOutput {
        exit_code,
        stdout,
        stderr: log::command_failed(flags, command, code, message),
    }
}

pub(crate) fn escape_json(s: &str) -> String {
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
