// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Repository automation for the Tessera workspaces, run as `cargo xtask <command>`.
//!
//! Commands are added by the work that needs them (RFC-0001 'Repository tree'): repository
//! policy checks (`lint`), the verifier isolation gate, vector generation and release helpers.
//! Every policy check also runs as a test of this crate, so `cargo test --workspace` enforces it.

#![forbid(unsafe_code)]
// A command-line tool reports on stdout and stderr.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod licenses;
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
  lint    check repository policies: unsafe code, SPDX headers, manifest licenses
  help    print this message
";

fn main() -> ExitCode {
    match std::env::args().nth(1).as_deref() {
        Some("help" | "-h" | "--help") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("lint") => match lint::run(&repo::root()) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
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
