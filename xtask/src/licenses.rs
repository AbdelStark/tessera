// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Manifest licenses (09 §7): every Cargo package and every Python project declares
//! `Apache-2.0 OR MIT`, either directly or, for Cargo, by inheriting it from its workspace.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use toml::{Table, Value};

use crate::spdx::LICENSE;

/// Checks every `Cargo.toml` and `pyproject.toml` in `files` (relative to `repo`).
pub(crate) fn check(repo: &Path, files: &[PathBuf]) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    for rel in files.iter().filter(|rel| !is_third_party_stand_in(rel)) {
        let problem = match rel.file_name().and_then(|n| n.to_str()) {
            Some("Cargo.toml") => cargo_problem(repo, rel)?,
            Some("pyproject.toml") => pyproject_problem(&parse(&repo.join(rel))?),
            _ => continue,
        };
        if let Some(problem) = problem {
            violations.push(format!("{}: {problem}", rel.display()));
        }
    }
    Ok(violations)
}

/// Fixture stand-ins for crates.io packages (`xtask/fixtures/<name>/vendor/`) model third-party
/// metadata, such as an MPL-2.0 license the dependency-policy self-test must reject, so their
/// manifests are not ours to license.
fn is_third_party_stand_in(rel: &Path) -> bool {
    let parts: Vec<_> = rel.components().map(|c| c.as_os_str()).collect();
    parts.len() > 4 && parts[0] == "xtask" && parts[1] == "fixtures" && parts[3] == "vendor"
}

fn parse(path: &Path) -> io::Result<Table> {
    let text = fs::read_to_string(path)?;
    text.parse::<Table>().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {e}", path.display()),
        )
    })
}

fn is_expected(value: Option<&Value>) -> bool {
    value.and_then(Value::as_str) == Some(LICENSE)
}

/// Why a Cargo manifest violates the policy, or `None`. Virtual manifests have no package and
/// are only checked through the packages that inherit from them.
fn cargo_problem(repo: &Path, rel: &Path) -> io::Result<Option<String>> {
    let manifest = parse(&repo.join(rel))?;
    let Some(package) = manifest.get("package").and_then(Value::as_table) else {
        return Ok(None);
    };
    let license = package.get("license");
    if is_expected(license) {
        return Ok(None);
    }
    let inherits = license
        .and_then(Value::as_table)
        .and_then(|t| t.get("workspace"))
        .and_then(Value::as_bool)
        == Some(true);
    if !inherits {
        return Ok(Some(format!(
            "[package] must declare `license = \"{LICENSE}\"`"
        )));
    }
    let dir = repo
        .join(rel)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let Some(root) = workspace_root(&dir, package)? else {
        return Ok(Some(
            "inherits its license but no workspace root was found".to_owned(),
        ));
    };
    let workspace_license = root
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("license"));
    if is_expected(workspace_license) {
        Ok(None)
    } else {
        Ok(Some(format!(
            "inherits its license, but its workspace root does not declare \
             `[workspace.package] license = \"{LICENSE}\"`"
        )))
    }
}

/// The manifest of the workspace a package belongs to: the one named by `package.workspace`, or
/// the nearest manifest with a `[workspace]` table at or above the package directory.
fn workspace_root(package_dir: &Path, package: &Table) -> io::Result<Option<Table>> {
    if let Some(explicit) = package.get("workspace").and_then(Value::as_str) {
        return parse(&package_dir.join(explicit).join("Cargo.toml")).map(Some);
    }
    for dir in package_dir.ancestors() {
        let candidate = dir.join("Cargo.toml");
        if candidate.is_file() {
            let manifest = parse(&candidate)?;
            if manifest.contains_key("workspace") {
                return Ok(Some(manifest));
            }
        }
    }
    Ok(None)
}

fn pyproject_problem(pyproject: &Table) -> Option<String> {
    let license = pyproject.get("project").and_then(|p| p.get("license"));
    (!is_expected(license)).then(|| format!("[project] must declare `license = \"{LICENSE}\"`"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Fixture;

    #[test]
    fn manifests_without_the_dual_license_are_reported() {
        let fixture = Fixture::new("licenses");
        let manifests = [
            // A workspace whose member inherits the license: both fine.
            (
                "ws/Cargo.toml",
                "[workspace]\n[workspace.package]\nlicense = \"Apache-2.0 OR MIT\"\n",
            ),
            (
                "ws/member/Cargo.toml",
                "[package]\nname = \"m\"\nlicense.workspace = true\n",
            ),
            // A package that states the license directly: fine.
            (
                "direct/Cargo.toml",
                "[package]\nname = \"d\"\nlicense = \"Apache-2.0 OR MIT\"\n",
            ),
            // Wrong expression.
            (
                "mit/Cargo.toml",
                "[package]\nname = \"x\"\nlicense = \"MIT\"\n",
            ),
            // No license at all.
            ("none/Cargo.toml", "[package]\nname = \"y\"\n"),
            // Inherits from a workspace that does not declare it.
            ("bare/Cargo.toml", "[workspace]\n"),
            (
                "bare/member/Cargo.toml",
                "[package]\nname = \"z\"\nlicense = { workspace = true }\n",
            ),
            (
                "py/pyproject.toml",
                "[project]\nname = \"p\"\nlicense = \"Apache-2.0 OR MIT\"\n",
            ),
            (
                "pybad/pyproject.toml",
                "[project]\nname = \"q\"\nlicense = { text = \"MIT\" }\n",
            ),
            // A third-party stand-in under xtask/fixtures/*/vendor/ is not checked.
            (
                "xtask/fixtures/x/vendor/dep-1.0.0/Cargo.toml",
                "[package]\nname = \"dep\"\nlicense = \"MPL-2.0\"\n",
            ),
        ];
        let paths: Vec<PathBuf> = manifests
            .iter()
            .map(|(name, text)| {
                fixture.write(name, text);
                PathBuf::from(name)
            })
            .collect();
        let violations = check(fixture.path(), &paths).expect("fixture is readable");
        let flagged: Vec<&str> = violations
            .iter()
            .map(|v| v.split(':').next().unwrap_or_default())
            .collect();
        assert_eq!(
            flagged,
            [
                "mit/Cargo.toml",
                "none/Cargo.toml",
                "bare/member/Cargo.toml",
                "pybad/pyproject.toml"
            ]
        );
    }
}
