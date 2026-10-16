// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask lint`: repository policies that are checked without building anything.

use std::io;
use std::path::Path;

use crate::{
    deny_policy, licenses, links, repo, reports, secret_access, spdx, unsafe_policy, versions,
};

/// The policies checked by `lint`, each with its violations.
pub(crate) fn checks(repo: &Path) -> io::Result<Vec<(&'static str, Vec<String>)>> {
    let files = repo::files(repo)?;
    Ok(vec![
        ("unsafe-code policy (RFC-0001)", unsafe_policy::check(repo)?),
        ("SPDX headers (09 §7)", spdx::check(repo, &files)?),
        ("manifest licenses (09 §7)", licenses::check(repo, &files)?),
        (
            "Markdown links and anchors (07 §9)",
            links::check(repo, &files)?,
        ),
        ("dependency-policy files (09 §7)", deny_policy::check(repo)?),
        (
            "secret access only in listed modules (05 §3 OBS-1)",
            secret_access::check(repo, &files)?,
        ),
        (
            "committed reports match their JSON Schemas (03 §5)",
            reports::check_reports(repo, &files)?,
        ),
        (
            "one version for the crates and tessera-quant (09 §1)",
            versions::check(repo)?,
        ),
    ])
}

/// Runs every check, reports on stdout and returns whether the repository is clean.
pub(crate) fn run(repo: &Path) -> io::Result<bool> {
    let mut clean = true;
    for (name, violations) in checks(repo)? {
        if violations.is_empty() {
            println!("ok    {name}");
        } else {
            clean = false;
            println!("FAIL  {name}");
            for violation in violations {
                println!("      {violation}");
            }
        }
    }
    Ok(clean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_passes_every_lint() {
        let failures: Vec<String> = checks(&repo::root())
            .expect("repository is readable")
            .into_iter()
            .flat_map(|(name, violations)| {
                violations.into_iter().map(move |v| format!("{name}: {v}"))
            })
            .collect();
        assert!(
            failures.is_empty(),
            "lint failures:\n{}",
            failures.join("\n")
        );
    }
}
