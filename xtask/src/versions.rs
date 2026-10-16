// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover release train's one version (09 §1): the workspace `Cargo.toml` and the
//! `tessera-quant` package's `python/pyproject.toml` state the same `MAJOR.MINOR.PATCH`.

use std::fs;
use std::io;
use std::path::Path;

use toml::{Table, Value};

/// The Python package's manifest, relative to the repository root.
const PYPROJECT: &str = "python/pyproject.toml";

/// The violations of the lockstep: a version missing, not plain `MAJOR.MINOR.PATCH`, or
/// different between the two manifests.
pub(crate) fn check(repo: &Path) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    let cargo = parse(repo, "Cargo.toml")?;
    let workspace = cargo
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(Value::as_str);
    let Some(workspace) = workspace else {
        violations.push("Cargo.toml: [workspace.package] must state the train's version".into());
        return Ok(violations);
    };
    if !repo.join(PYPROJECT).is_file() {
        violations.push(format!(
            "{PYPROJECT} is missing: the prover release train includes tessera-quant"
        ));
        return Ok(violations);
    }
    let pyproject = parse(repo, PYPROJECT)?;
    let project = pyproject.get("project");
    let dynamic = project
        .and_then(|p| p.get("dynamic"))
        .and_then(Value::as_array)
        .is_some_and(|fields| fields.iter().any(|f| f.as_str() == Some("version")));
    let python = project
        .and_then(|p| p.get("version"))
        .and_then(Value::as_str);
    let Some(python) = python.filter(|_| !dynamic) else {
        violations.push(format!(
            "{PYPROJECT}: [project] must state a static version"
        ));
        return Ok(violations);
    };
    for (file, version) in [("Cargo.toml", workspace), (PYPROJECT, python)] {
        if !is_plain(version) {
            violations.push(format!(
                "{file}: version `{version}` is not MAJOR.MINOR.PATCH, and 09 §1 maps no \
                 pre-release or build suffix between SemVer and PEP 440"
            ));
        }
    }
    if violations.is_empty() && workspace != python {
        violations.push(format!(
            "Cargo.toml says {workspace} and {PYPROJECT} says {python}: the prover release train \
             has one version"
        ));
    }
    Ok(violations)
}

/// `MAJOR.MINOR.PATCH` in decimal without leading zeros: the form SemVer and PEP 440 share.
fn is_plain(version: &str) -> bool {
    let parts: Vec<&str> = version.split('.').collect();
    parts.len() == 3
        && parts.iter().all(|part| {
            !part.is_empty()
                && part.bytes().all(|b| b.is_ascii_digit())
                && (*part == "0" || !part.starts_with('0'))
        })
}

fn parse(repo: &Path, rel: &str) -> io::Result<Table> {
    let text = fs::read_to_string(repo.join(rel))?;
    text.parse::<Table>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{rel}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Fixture;

    fn violations(cargo: &str, pyproject: Option<&str>) -> Vec<String> {
        let fixture = Fixture::new("versions");
        fixture.write("Cargo.toml", cargo);
        if let Some(pyproject) = pyproject {
            fixture.write(PYPROJECT, pyproject);
        }
        check(fixture.path()).expect("the fixture is readable")
    }

    fn workspace(version: &str) -> String {
        format!("[workspace]\n[workspace.package]\nversion = \"{version}\"\n")
    }

    fn project(version: &str) -> String {
        format!("[project]\nname = \"tessera-quant\"\nversion = \"{version}\"\n")
    }

    #[test]
    fn equal_plain_versions_pass() {
        assert_eq!(
            violations(&workspace("0.1.0"), Some(&project("0.1.0"))),
            Vec::<String>::new()
        );
        assert_eq!(
            violations(&workspace("10.20.30"), Some(&project("10.20.30"))),
            Vec::<String>::new()
        );
    }

    #[test]
    fn different_versions_fail_naming_both() {
        let found = violations(&workspace("0.2.0"), Some(&project("0.1.0")));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].contains("0.2.0") && found[0].contains("0.1.0"),
            "{found:?}"
        );
    }

    #[test]
    fn a_missing_or_unstated_version_fails() {
        assert_eq!(
            violations(&workspace("0.1.0"), None),
            ["python/pyproject.toml is missing: the prover release train includes tessera-quant"]
        );
        assert_eq!(
            violations(
                &workspace("0.1.0"),
                Some("[project]\nname = \"tessera-quant\"\ndynamic = [\"version\"]\n")
            ),
            ["python/pyproject.toml: [project] must state a static version"]
        );
        // A dynamic version is not the static one, even when both are written down.
        assert_eq!(
            violations(
                &workspace("0.1.0"),
                Some(&format!("{}dynamic = [\"version\"]\n", project("0.1.0")))
            )
            .len(),
            1
        );
        assert_eq!(
            violations("[workspace]\n", Some(&project("0.1.0"))),
            ["Cargo.toml: [workspace.package] must state the train's version"]
        );
    }

    #[test]
    fn versions_outside_the_shared_form_fail_even_when_equal() {
        for version in [
            "0.2.0-rc.1",
            "0.2.0rc1",
            "0.1.0+build",
            "0.01.0",
            "1.0",
            "1.0.0.0",
            "v1.0.0",
        ] {
            let found = violations(&workspace(version), Some(&project(version)));
            assert_eq!(found.len(), 2, "{version}: {found:?}");
            assert!(
                found.iter().all(|v| v.contains("MAJOR.MINOR.PATCH")),
                "{found:?}"
            );
        }
    }
}
