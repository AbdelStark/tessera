// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `tessera-verify` command-line interface and its JSON verdict report (RFC-0019, 02 §2.3).
//!
//! Currently exposes the `params` subcommand (`tessera-verify params`), which runs the verifier's
//! independent soundness calculator ([`tv_core::soundness`], RFC-0014 'Soundness calculator').

#![forbid(unsafe_code)]

use tv_core::soundness::{
    DEFAULT_EXP_LOOKUPS, DEFAULT_EXP_ROWS, LookupSize, SoundnessParams, SoundnessReport,
    StatementSizes,
};

/// Exit code on success (`02 §2.4`).
pub const EXIT_SUCCESS: i32 = 0;

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
    params    Compute the independent soundness table for a parameter set (RFC-0014)
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
    let collected: Vec<String> = args.into_iter().map(|s| String::from(s.as_ref())).collect();
    let json_flag = collected.iter().any(|a| a == "--json");

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
        other => usage_error(
            "tessera-verify",
            &format!("unknown command `{other}`"),
            json_flag,
        ),
    }
}

fn run_params(args: &[String], json_flag: bool) -> CommandOutput {
    if args.iter().any(|a| a == "-h" || a == "--help") {
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
            let stdout = if json_flag {
                report.to_json()
            } else {
                report.to_table()
            };
            CommandOutput {
                exit_code: EXIT_SUCCESS,
                stdout,
                stderr: String::new(),
            }
        }
        Err(err) => usage_error("params", &err.to_string(), json_flag),
    }
}

fn usage_error(command: &str, message: &str, json_flag: bool) -> CommandOutput {
    if json_flag {
        let escaped = escape_json(message);
        let stdout = format!(
            "{{\"command\":\"{command}\",\"error\":{{\"code\":\"ERR-USAGE\",\"message\":\"{escaped}\"}}}}\n"
        );
        CommandOutput {
            exit_code: EXIT_USAGE,
            stdout,
            stderr: String::new(),
        }
    } else {
        CommandOutput {
            exit_code: EXIT_USAGE,
            stdout: String::new(),
            stderr: format!("error (ERR-USAGE): {message}\n"),
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
            c => out.push(c),
        }
    }
    out
}
