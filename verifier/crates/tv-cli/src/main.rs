// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Binary entry point for `tessera-verify` (RFC-0019, 02 §2.3).

#![forbid(unsafe_code)]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let output = tv_cli::run(env::args().skip(1));
    if !output.stdout.is_empty() {
        print!("{}", output.stdout);
    }
    if !output.stderr.is_empty() {
        eprint!("{}", output.stderr);
    }
    let code = u8::try_from(output.exit_code).unwrap_or(70);
    ExitCode::from(code)
}
