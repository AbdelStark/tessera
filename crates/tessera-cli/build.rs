// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Records the build's git commit and rustc version for the run manifest (05 §6). Either is
//! empty when it cannot be read, as in a build outside a git checkout.

use std::path::Path;
use std::process::Command;

fn output(program: &str, args: &[&str]) -> Option<String> {
    let dir = std::env::var_os("CARGO_MANIFEST_DIR")?;
    let out = Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let version = output(&rustc, &["--version"]).unwrap_or_default();
    println!("cargo:rustc-env=TESSERA_RUSTC_VERSION={version}");
    let commit = std::env::var("TESSERA_GIT_COMMIT")
        .ok()
        .or_else(|| output("git", &["rev-parse", "HEAD"]))
        .unwrap_or_default();
    println!("cargo:rustc-env=TESSERA_GIT_COMMIT={commit}");
    println!("cargo:rerun-if-env-changed=TESSERA_GIT_COMMIT");
    // Rebuild when HEAD moves: HEAD itself, and the branch file it names.
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let mut watched = vec![output("git", &["rev-parse", "--git-path", "HEAD"])];
    if let Some(branch) = output("git", &["symbolic-ref", "-q", "HEAD"]) {
        watched.push(output("git", &["rev-parse", "--git-path", &branch]));
        watched.push(output("git", &["rev-parse", "--git-path", "packed-refs"]));
    }
    for path in watched.into_iter().flatten() {
        let path = Path::new(&dir).join(path);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
