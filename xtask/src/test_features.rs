// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask check-test-features`: the test-only features of RFC-0001 'Cargo features'
//! (`test-rng`, `test-params`, `unsafe-debug-dump`) cannot reach a release build.
//!
//! Every crate in either workspace that declares one of these features is found through
//! `cargo metadata`, so a crate that adds such a feature later is covered without a list to
//! update. For each (crate, feature): `cargo check --release` must fail with the crate's
//! `compile_error!` guard, and a debug check must succeed. `compile_error!` is emitted while
//! the crate is expanded, before code generation, so `check` fails exactly where `build` would.

use std::ffi::OsString;
use std::io;
use std::path::Path;
use std::process::Command;

use serde_json::Value as Json;

/// Features that weaken security and must never be enabled in a release build.
pub(crate) const TEST_ONLY: [&str; 3] = ["test-rng", "test-params", "unsafe-debug-dump"];

/// Text every guard's message contains, so a release check is seen to fail for this reason
/// and not for an unrelated compile error.
pub(crate) const GUARD_MARKER: &str = "must not be enabled in a release build";

/// One crate that declares a test-only feature.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Gated {
    pub(crate) workspace: &'static str,
    pub(crate) package: String,
    pub(crate) feature: &'static str,
}

fn cargo(dir: &Path, toolchain: Option<&str>) -> Command {
    let program: OsString = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(program);
    cmd.current_dir(dir);
    if let Some(channel) = toolchain {
        cmd.env("RUSTUP_TOOLCHAIN", channel);
    }
    cmd
}

/// Every (workspace, package, test-only feature) declared in the workspaces under `repo`.
pub(crate) fn discover(repo: &Path, workspaces: &[&'static str]) -> io::Result<Vec<Gated>> {
    let mut found = Vec::new();
    for &workspace in workspaces {
        let output = cargo(&repo.join(workspace), None)
            .args(["metadata", "--format-version", "1", "--no-deps", "--locked"])
            .output()?;
        if !output.status.success() {
            return Err(io::Error::other(format!(
                "cargo metadata in {workspace}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        let metadata: Json = serde_json::from_slice(&output.stdout).map_err(io::Error::other)?;
        for package in metadata["packages"].as_array().into_iter().flatten() {
            let Some(features) = package["features"].as_object() else {
                continue;
            };
            for feature in TEST_ONLY {
                if features.contains_key(feature) {
                    found.push(Gated {
                        workspace,
                        package: package["name"].as_str().unwrap_or_default().to_owned(),
                        feature,
                    });
                }
            }
        }
    }
    found.sort_by(|a, b| {
        (a.workspace, &a.package, a.feature).cmp(&(b.workspace, &b.package, b.feature))
    });
    Ok(found)
}

/// Checks every discovered (crate, feature) pair and returns one message per violation.
pub(crate) fn check(
    repo: &Path,
    workspaces: &[&'static str],
    toolchain: Option<&str>,
) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    for gated in discover(repo, workspaces)? {
        let dir = repo.join(gated.workspace);
        let id = format!("{} --features {}", gated.package, gated.feature);
        let release = cargo(&dir, toolchain)
            .args([
                "check",
                "--locked",
                "--release",
                "-p",
                &gated.package,
                "--features",
                gated.feature,
            ])
            .output()?;
        let stderr = String::from_utf8_lossy(&release.stderr);
        if release.status.success() {
            violations.push(format!(
                "{id}: a release build compiles; add the compile_error! guard"
            ));
        } else if !stderr.contains(GUARD_MARKER) {
            violations.push(format!(
                "{id}: the release build fails, but not on the guard: {}",
                stderr.trim()
            ));
        }
        let debug = cargo(&dir, toolchain)
            .args([
                "check",
                "--locked",
                "-p",
                &gated.package,
                "--features",
                gated.feature,
            ])
            .output()?;
        if !debug.status.success() {
            violations.push(format!(
                "{id}: a debug build fails: {}",
                String::from_utf8_lossy(&debug.stderr).trim()
            ));
        }
    }
    Ok(violations)
}

/// Runs the check on both workspaces of the repository and reports on stdout.
pub(crate) fn run(repo: &Path) -> io::Result<bool> {
    let toolchain = crate::repo::pinned_channel(&repo.join("rust-toolchain.toml"));
    let gated = discover(repo, &[".", "verifier"])?;
    for g in &gated {
        println!(
            "      {:<10} {} --features {}",
            g.workspace, g.package, g.feature
        );
    }
    let violations = check(repo, &[".", "verifier"], toolchain.as_deref())?;
    if violations.is_empty() {
        println!(
            "ok    {} test-only (crate, feature) pairs fail release builds and pass debug builds",
            gated.len()
        );
        return Ok(true);
    }
    println!("FAIL  test-only features (RFC-0001 'Cargo features')");
    for violation in violations {
        println!("      {violation}");
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Fixture;

    fn package(fixture: &Fixture, name: &str, feature: &str, lib: &str) {
        fixture.write(
            &format!("{name}/Cargo.toml"),
            &format!(
                "[package]\nname = \"{name}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[features]\n{feature} = []\n"
            ),
        );
        fixture.write(&format!("{name}/src/lib.rs"), lib);
    }

    #[test]
    fn missing_wrong_and_overbroad_guards_are_reported() {
        let fixture = Fixture::new("test-features");
        fixture.write(
            "Cargo.toml",
            "[workspace]\nresolver = \"3\"\nmembers = [\"guarded\", \"unguarded\", \"other\", \"always\"]\n",
        );
        let guard = |feature: &str, message: &str| {
            format!(
                "#[cfg(all(feature = \"{feature}\", not(debug_assertions)))]\ncompile_error!(\"{message}\");\n"
            )
        };
        package(
            &fixture,
            "guarded",
            "test-rng",
            &guard("test-rng", "x must not be enabled in a release build"),
        );
        package(&fixture, "unguarded", "test-params", "");
        package(
            &fixture,
            "other",
            "unsafe-debug-dump",
            &guard("unsafe-debug-dump", "an unrelated error"),
        );
        package(
            &fixture,
            "always",
            "test-rng",
            "#[cfg(feature = \"test-rng\")]\ncompile_error!(\"must not be enabled in a release build\");\n",
        );
        let pin = crate::repo::pinned_channel(&crate::repo::root().join("rust-toolchain.toml"));
        let lock = cargo(fixture.path(), pin.as_deref())
            .arg("generate-lockfile")
            .output()
            .expect("run cargo");
        assert!(
            lock.status.success(),
            "{}",
            String::from_utf8_lossy(&lock.stderr)
        );

        let found = discover(fixture.path(), &["."]).expect("metadata");
        assert_eq!(found.len(), 4, "{found:#?}");
        let violations = check(fixture.path(), &["."], pin.as_deref()).expect("checks run");
        assert_eq!(violations.len(), 3, "{violations:#?}");
        assert!(violations[0].starts_with("always --features test-rng: a debug build fails"));
        assert!(violations[1].starts_with(
            "other --features unsafe-debug-dump: the release build fails, but not on the guard"
        ));
        assert!(
            violations[2].starts_with("unguarded --features test-params: a release build compiles")
        );
    }
}
