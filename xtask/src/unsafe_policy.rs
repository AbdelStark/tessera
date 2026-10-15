// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The unsafe-code policy of RFC-0001 'Language boundaries': every prover crate declares
//! `#![forbid(unsafe_code)]` at each crate root, except the crates the RFC names, which must
//! still deny it crate-wide so that each unsafe module opts in explicitly.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Crates that RFC-0001 permits to contain unsafe code: SIMD field backends and CUDA FFI.
pub(crate) const UNSAFE_PERMITTED: [&str; 2] = ["tessera-field", "tessera-gpu"];

const FORBID: &str = "#![forbid(unsafe_code)]";
const DENY: &str = "#![deny(unsafe_code)]";

/// Checks every crate under `<repo>/crates` and returns one message per violation.
pub(crate) fn check(repo: &Path) -> io::Result<Vec<String>> {
    let mut crate_dirs: Vec<PathBuf> = fs::read_dir(repo.join("crates"))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<io::Result<_>>()?;
    crate_dirs.retain(|dir| dir.join("Cargo.toml").is_file());
    crate_dirs.sort();

    let mut violations = Vec::new();
    for dir in &crate_dirs {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let roots = crate_roots(dir)?;
        if roots.is_empty() {
            violations.push(format!(
                "{name}: no crate root (src/lib.rs, src/main.rs or src/bin/*.rs)"
            ));
        }
        for root in roots {
            let text = fs::read_to_string(&root)?;
            let has = |attr: &str| text.lines().any(|line| line.trim() == attr);
            let shown = root.strip_prefix(repo).unwrap_or(&root).display();
            if UNSAFE_PERMITTED.contains(&name.as_str()) {
                if !has(FORBID) && !has(DENY) {
                    violations.push(format!("{shown}: must declare `{DENY}` or `{FORBID}`"));
                }
            } else if !has(FORBID) {
                violations.push(format!("{shown}: must declare `{FORBID}`"));
            }
        }
    }
    Ok(violations)
}

/// The library, binary and `src/bin/*.rs` roots of one crate, in a stable order.
fn crate_roots(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let src = dir.join("src");
    let mut roots: Vec<PathBuf> = ["lib.rs", "main.rs"]
        .iter()
        .map(|f| src.join(f))
        .filter(|p| p.is_file())
        .collect();
    let bin = src.join("bin");
    if bin.is_dir() {
        let mut bins: Vec<PathBuf> = fs::read_dir(&bin)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<io::Result<_>>()?;
        bins.retain(|p| p.extension().is_some_and(|ext| ext == "rs"));
        bins.sort();
        roots.extend(bins);
    }
    Ok(roots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{repo, testing::Fixture};

    #[test]
    fn exempt_crates_exist() {
        // A stale exemption would silently exempt nothing; a renamed crate must update the list.
        for name in UNSAFE_PERMITTED {
            assert!(
                repo::root().join("crates").join(name).is_dir(),
                "exempt crate {name} is missing"
            );
        }
    }

    #[test]
    fn missing_or_weakened_attributes_are_reported() {
        let fixture = Fixture::new("unsafe");
        fixture.write("crates/tessera-ok/Cargo.toml", "");
        fixture.write(
            "crates/tessera-ok/src/lib.rs",
            "//! ok\n\n#![forbid(unsafe_code)]\n",
        );
        fixture.write("crates/tessera-plain/Cargo.toml", "");
        fixture.write("crates/tessera-plain/src/lib.rs", "//! no attribute\n");
        fixture.write("crates/tessera-bin/Cargo.toml", "");
        fixture.write("crates/tessera-bin/src/lib.rs", "#![forbid(unsafe_code)]\n");
        fixture.write(
            "crates/tessera-bin/src/bin/tool.rs",
            "// forbid(unsafe_code) in a comment\n",
        );
        fixture.write("crates/tessera-gpu/Cargo.toml", "");
        fixture.write("crates/tessera-gpu/src/lib.rs", "#![allow(unsafe_code)]\n");

        let violations = check(fixture.path()).expect("fixture is readable");
        let flagged: Vec<&str> = violations
            .iter()
            .map(|v| v.split(':').next().unwrap_or_default())
            .collect();
        assert_eq!(
            flagged,
            [
                "crates/tessera-bin/src/bin/tool.rs",
                "crates/tessera-gpu/src/lib.rs",
                "crates/tessera-plain/src/lib.rs",
            ]
        );
    }
}
