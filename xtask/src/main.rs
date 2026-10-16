// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Repository automation for the Tessera workspaces, run as `cargo xtask <command>`.
//!
//! Commands are added by the work that needs them (RFC-0001 'Repository tree'): repository
//! policy checks (`lint`), the verifier isolation gate, vector generation and release helpers.
//! Every policy check also runs as a test of this crate, so `cargo test --workspace` enforces it.

#![forbid(unsafe_code)]
// A command-line tool reports on stdout and stderr.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod branchless;
mod deny_policy;
mod deps;
mod isolation;
mod licenses;
mod links;
mod lint;
mod pr_policy;
mod repo;
mod secret_access;
mod spdx;
mod test_features;
#[cfg(test)]
mod testing;
mod unsafe_policy;
mod vectors;

use std::process::ExitCode;

/// Exit code for usage errors, shared with the Tessera command-line tools (02 §2.4).
const EXIT_USAGE: u8 = 64;

const USAGE: &str = "\
usage: cargo xtask <command>

commands:
  check-branchless   check that the field's operations compile branch-free in x86_64 loops
                     (INV-ARCH-2; needs the x86_64-unknown-linux-gnu standard library)
  check-deps         run cargo-deny in both workspaces and cargo-vet in the verifier, then
                     self-test the dependency policies (09 §7; needs cargo-deny, cargo-vet)
  check-isolation    check that the verifier shares no code with the prover (RFC-0001)
  check-pr --base <REF> [--labels <a,b,...>]
                     check the change from REF to HEAD: changelog entries and DCO sign-off
  check-test-features
                     check that test-only cargo features fail release builds (RFC-0001)
  gen-vectors [--check]
                     write the conformance vectors in spec/vectors/: the codec vectors, and
                     the files of each set's Python oracle (generate.py; needs python3);
                     or check that they are current, running the oracles on a copy
  lint               check repository policies: unsafe code, SPDX headers, manifest
                     licenses, Markdown links, dependency-policy files
  help               print this message
";

fn main() -> ExitCode {
    match std::env::args().nth(1).as_deref() {
        Some("help" | "-h" | "--help") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("check-branchless") => report(branchless::run(&repo::root())),
        Some("check-deps") => report(deps::run(&repo::root())),
        Some("check-isolation") => report(isolation::run(&repo::root())),
        Some("check-pr") => match check_pr_args(std::env::args().skip(2)) {
            Ok((base, labels)) => report(pr_policy::run(&repo::root(), &base, &labels)),
            Err(message) => {
                eprintln!("error: {message}\n\n{USAGE}");
                ExitCode::from(EXIT_USAGE)
            }
        },
        Some("check-test-features") => report(test_features::run(&repo::root())),
        Some("gen-vectors") => match std::env::args().nth(2).as_deref() {
            None => report(vectors::run(&repo::root(), false)),
            Some("--check") => report(vectors::run(&repo::root(), true)),
            Some(other) => {
                eprintln!("gen-vectors: unknown argument `{other}`\n\n{USAGE}");
                ExitCode::from(EXIT_USAGE)
            }
        },
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

/// Parses `--base <REF> [--labels <a,b,...>]`.
fn check_pr_args(mut args: impl Iterator<Item = String>) -> Result<(String, Vec<String>), String> {
    let (mut base, mut labels) = (None, Vec::new());
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("`{arg}` needs a value"))?;
        match arg.as_str() {
            "--base" => base = Some(value),
            "--labels" => {
                labels = value
                    .split(',')
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(str::to_owned)
                    .collect();
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok((base.ok_or("check-pr needs --base <REF>")?, labels))
}
