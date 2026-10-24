// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Capture public build metadata without runtime tool invocations.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for name in ["HEAD", "refs/heads", "packed-refs"] {
        if let Ok(output) = Command::new("git")
            .args(["rev-parse", "--git-path", name])
            .output()
            && output.status.success()
        {
            println!(
                "cargo:rerun-if-changed={}",
                String::from_utf8_lossy(&output.stdout).trim()
            );
        }
    }
    if let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output()
        && output.status.success()
    {
        println!(
            "cargo:rustc-env=TESSERA_VERIFIER_BUILD_COMMIT={}",
            String::from_utf8_lossy(&output.stdout).trim()
        );
    }
    if let Ok(output) = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .arg("--version")
        .output()
        && output.status.success()
    {
        println!(
            "cargo:rustc-env=TESSERA_VERIFIER_BUILD_RUSTC={}",
            String::from_utf8_lossy(&output.stdout).trim()
        );
    }
}
