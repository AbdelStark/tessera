// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Repository automation for the Tessera workspaces, run as `cargo xtask <command>`.
//!
//! Commands are added by the work that needs them (RFC-0001 'Repository tree'): repository
//! policy checks (`lint`), the verifier isolation gate, vector generation and release helpers.
//! Every policy check also runs as a test of this crate, so `cargo test --workspace` enforces it.

#![forbid(unsafe_code)]
// A command-line tool reports on stdout and stderr.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod isolation;
mod licenses;
mod links;
mod lint;
mod repo;
mod spdx;
#[cfg(test)]
mod testing;
mod unsafe_policy;

use std::process::ExitCode;

/// Exit code for usage errors, shared with the Tessera command-line tools (02 §2.4).
const EXIT_USAGE: u8 = 64;

const USAGE: &str = "\
usage: cargo xtask <command>

commands:
  check-isolation    check that the verifier shares no code with the prover (RFC-0001)
  lint               check repository policies: unsafe code, SPDX headers, manifest
                     licenses, Markdown links
  help               print this message
";

fn main() -> ExitCode {
    match std::env::args().nth(1).as_deref() {
        Some("help" | "-h" | "--help") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("check-isolation") => report(isolation::run(&repo::root())),
        Some("lint") => report(lint::run(&repo::root())),
        Some(other) => {
            eprintln!("error: unknown command `{other}`\n\n{USAGE}");
            ExitCode::from(EXIT_USAGE)
        }
        None => {
            eprint!("{USAGE}");
            ExitCode::from(EXIT_USAGE)
        }
    }
}

/// Maps a check's outcome to the process exit code: 0 clean, 1 violations or errors.
fn report(outcome: std::io::Result<bool>) -> ExitCode {
    match outcome {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
