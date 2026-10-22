// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Binary entry point for `tessera-verify` (RFC-0019, 02 §2.3).

#![forbid(unsafe_code)]

use std::env;
use std::io::{self, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    // Panic payloads can contain protocol values; only the public ERR-INTERNAL is reported.
    std::panic::set_hook(Box::new(|_| {}));
    let args = env::args_os()
        .skip(1)
        .take(65)
        .map(|a| a.into_string())
        .collect::<Result<Vec<_>, _>>();
    let output = match args {
        Ok(args) => tv_cli::run(args),
        Err(_) => tv_cli::run(["invalid-utf8-argument"]),
    };
    if io::stdout()
        .lock()
        .write_all(output.stdout.as_bytes())
        .is_err()
        || io::stderr()
            .lock()
            .write_all(output.stderr.as_bytes())
            .is_err()
    {
        return ExitCode::from(20);
    }
    ExitCode::from(u8::try_from(output.exit_code).unwrap_or(70))
}
