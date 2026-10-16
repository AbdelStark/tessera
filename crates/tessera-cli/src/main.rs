// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `tessera` binary: [`tessera_cli::run`] on the process's arguments.

#![forbid(unsafe_code)]

fn main() -> std::process::ExitCode {
    tessera_cli::run(std::env::args_os())
}
