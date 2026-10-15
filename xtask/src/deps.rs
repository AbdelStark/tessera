// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask check-deps`: the dependency policy (09 §7; trust boundary TB-4 in 06 §3).
//!
//! 1. `cargo deny check` in both workspaces: licenses, advisories, bans and sources.
//! 2. `cargo vet --locked` in the verifier workspace: every third-party crate audited.
//! 3. A self-test of the cargo-deny policies on `xtask/fixtures/deny`, so that a weakened policy
//!    fails here even while the real graph happens to pass: the verifier and prover policies
//!    reject an MPL-2.0 crate, the prover policy accepts it once an exception with a reason is
//!    added, and the advisory check rejects a crate covered by a RustSec advisory.
//!
//! Needs `cargo-deny` and `cargo-vet` on `PATH`; CI installs pinned versions.

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::Path;
use std::process::{Command, Output};

use crate::repo::copy_dir;

fn cargo(dir: &Path, args: &[&str]) -> io::Result<Output> {
    let program: OsString = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    Command::new(program).args(args).current_dir(dir).output()
}

/// `Ok` when the command succeeded, else its output for the report.
fn succeeded(output: Output) -> Result<(), String> {
    if output.status.success() {
        return Ok(());
    }
    let text = String::from_utf8_lossy(&output.stderr).into_owned()
        + &String::from_utf8_lossy(&output.stdout);
    let tail: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    Err(tail[tail.len().saturating_sub(12)..].join("\n"))
}

/// `Ok` when the command failed and its output contains `expected`.
fn rejected(output: Output, expected: &str) -> Result<(), String> {
    let text = String::from_utf8_lossy(&output.stderr).into_owned()
        + &String::from_utf8_lossy(&output.stdout);
    match (output.status.success(), text.contains(expected)) {
        (false, true) => Ok(()),
        (true, _) => Err("the check passed; the policy no longer rejects this input".to_owned()),
        (false, false) => Err(format!(
            "the check failed, but not with `{expected}`:\n{}",
            text.trim()
        )),
    }
}

/// Runs every check and the self-test; returns `(name, outcome)` pairs.
pub(crate) fn checks(repo: &Path) -> io::Result<Vec<(String, Result<(), String>)>> {
    let deny = |dir: &Path, config: &Path, which: &str| {
        let config = config.to_string_lossy().into_owned();
        cargo(dir, &["deny", "--config", &config, "check", which])
    };
    let mut results = Vec::new();
    for (name, dir) in [
        ("prover workspace", repo.to_path_buf()),
        ("verifier workspace", repo.join("verifier")),
    ] {
        let config = dir.join("deny.toml");
        let config = config.to_string_lossy().into_owned();
        let outcome = succeeded(cargo(&dir, &["deny", "--config", &config, "check"])?);
        results.push((format!("cargo deny check ({name})"), outcome));
    }
    results.push((
        "cargo vet --locked (verifier workspace)".to_owned(),
        succeeded(cargo(&repo.join("verifier"), &["vet", "--locked"])?),
    ));

    // Self-test on a temporary copy of the fixture.
    let tmp = std::env::temp_dir().join(format!("tessera-check-deps-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    copy_dir(&repo.join("xtask/fixtures/deny"), &tmp)?;
    for workspace in ["licensed", "vulnerable"] {
        succeeded(cargo(&tmp.join(workspace), &["generate-lockfile"])?)
            .map_err(|e| io::Error::other(format!("fixture lockfile for {workspace}: {e}")))?;
    }
    let prover_policy = repo.join("deny.toml");
    let verifier_policy = repo.join("verifier/deny.toml");
    let licensed = tmp.join("licensed");
    results.push((
        "self-test: the verifier policy rejects an MPL-2.0 crate".to_owned(),
        rejected(deny(&licensed, &verifier_policy, "licenses")?, "MPL-2.0"),
    ));
    results.push((
        "self-test: the prover policy rejects an MPL-2.0 crate without an exception".to_owned(),
        rejected(deny(&licensed, &prover_policy, "licenses")?, "MPL-2.0"),
    ));
    let with_exception = tmp.join("deny-with-exception.toml");
    fs::write(
        &with_exception,
        fs::read_to_string(&prover_policy)?
            + "\n# Reason: self-test exception from `cargo xtask check-deps`.\n\
               [[licenses.exceptions]]\ncrate = \"mpl-licensed\"\nallow = [\"MPL-2.0\"]\n",
    )?;
    results.push((
        "self-test: the prover policy accepts it with a recorded exception".to_owned(),
        succeeded(deny(&licensed, &with_exception, "licenses")?),
    ));
    results.push((
        "self-test: the advisory check rejects a crate under RUSTSEC-2021-0003".to_owned(),
        rejected(
            deny(&tmp.join("vulnerable"), &prover_policy, "advisories")?,
            "RUSTSEC-2021-0003",
        ),
    ));
    let _ = fs::remove_dir_all(&tmp);
    Ok(results)
}

/// Runs the checks and reports on stdout.
pub(crate) fn run(repo: &Path) -> io::Result<bool> {
    let mut clean = true;
    for (name, outcome) in checks(repo)? {
        match outcome {
            Ok(()) => println!("ok    {name}"),
            Err(detail) => {
                clean = false;
                println!("FAIL  {name}");
                for line in detail.lines() {
                    println!("      {line}");
                }
            }
        }
    }
    Ok(clean)
}
