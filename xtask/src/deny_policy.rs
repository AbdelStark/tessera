// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The dependency-policy files themselves (09 §7): both `deny.toml` files allow exactly the
//! licenses of 09 §7; every prover exception and ignored advisory states its reason; the
//! verifier admits neither. cargo-deny cannot express "with a reason" for license exceptions,
//! so a `# Reason:` comment above each `[[licenses.exceptions]]` table carries it.

use std::fs;
use std::io;
use std::path::Path;

use toml::{Table, Value};

/// The licenses 09 §7 allows, in any order.
pub(crate) const ALLOWED_LICENSES: [&str; 8] = [
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Zlib",
    "Unicode-3.0",
    "CC0-1.0",
];

const EXCEPTION_HEADER: &str = "[[licenses.exceptions]]";

/// Checks `deny.toml` (prover policy) and `verifier/deny.toml` under `repo`.
pub(crate) fn check(repo: &Path) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    for (rel, verifier) in [("deny.toml", false), ("verifier/deny.toml", true)] {
        let text = fs::read_to_string(repo.join(rel))?;
        violations.extend(
            policy_violations(&text, verifier)
                .into_iter()
                .map(|v| format!("{rel}: {v}")),
        );
    }
    Ok(violations)
}

/// Violations of one policy file's text; `verifier` selects the stricter rules.
pub(crate) fn policy_violations(text: &str, verifier: bool) -> Vec<String> {
    let table = match text.parse::<Table>() {
        Ok(table) => table,
        Err(e) => return vec![format!("does not parse: {e}")],
    };
    let mut violations = Vec::new();
    let licenses = table.get("licenses");

    let mut allow: Vec<&str> = licenses
        .and_then(|l| l.get("allow"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    allow.sort_unstable();
    let mut expected = ALLOWED_LICENSES.to_vec();
    expected.sort_unstable();
    if allow != expected {
        violations.push(format!(
            "licenses.allow must be exactly the 09 §7 list {expected:?}, found {allow:?}"
        ));
    }

    let exceptions = licenses
        .and_then(|l| l.get("exceptions"))
        .and_then(Value::as_array);
    let exception_count = exceptions.map_or(0, Vec::len);
    if verifier {
        if exception_count > 0 {
            violations.push("the verifier admits no license exceptions (09 §7)".to_owned());
        }
    } else {
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        let headers: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| **l == EXCEPTION_HEADER)
            .map(|(i, _)| i)
            .collect();
        if headers.len() != exception_count {
            violations.push(format!(
                "license exceptions must be `{EXCEPTION_HEADER}` tables (found {exception_count} \
                 exceptions, {} tables)",
                headers.len()
            ));
        }
        for i in headers {
            let reason = lines[..i]
                .iter()
                .rev()
                .find(|l| !l.is_empty())
                .and_then(|l| l.strip_prefix("# Reason:"))
                .map(str::trim);
            if reason.is_none_or(str::is_empty) {
                violations.push(format!(
                    "the exception at line {} needs a `# Reason: ...` comment directly above it",
                    i + 1
                ));
            }
        }
    }

    let ignores = table
        .get("advisories")
        .and_then(|a| a.get("ignore"))
        .and_then(Value::as_array);
    for entry in ignores.into_iter().flatten() {
        let id = entry
            .get("id")
            .and_then(Value::as_str)
            .or_else(|| entry.as_str())
            .unwrap_or("?");
        if verifier {
            violations.push(format!("the verifier ignores no advisory, found {id}"));
        } else if entry
            .get("reason")
            .and_then(Value::as_str)
            .is_none_or(|r| r.trim().is_empty())
        {
            violations.push(format!("ignored advisory {id} needs `reason = \"...\"`"));
        }
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALLOW: &str = "allow = [\"MIT\", \"Apache-2.0\", \"BSD-2-Clause\", \"BSD-3-Clause\", \"ISC\", \
                         \"Zlib\", \"Unicode-3.0\", \"CC0-1.0\"]\n";

    #[test]
    fn repository_policies_pass() {
        let violations = check(&crate::repo::root()).expect("policy files are readable");
        assert!(violations.is_empty(), "{violations:#?}");
    }

    #[test]
    fn reasons_and_allow_lists_are_enforced() {
        let with_reason = format!(
            "[licenses]\n{ALLOW}\n# Reason: needed for X.\n{EXCEPTION_HEADER}\ncrate = \"a\"\nallow = [\"MPL-2.0\"]\n"
        );
        assert!(policy_violations(&with_reason, false).is_empty());
        assert_eq!(
            policy_violations(&with_reason, true).len(),
            1,
            "the verifier admits none"
        );

        let without_reason = format!(
            "[licenses]\n{ALLOW}\n# Needed.\n{EXCEPTION_HEADER}\ncrate = \"a\"\nallow = [\"MPL-2.0\"]\n"
        );
        assert!(
            policy_violations(&without_reason, false)[0]
                .contains("needs a `# Reason: ...` comment")
        );

        let inline = format!(
            "[licenses]\n{ALLOW}exceptions = [{{ crate = \"a\", allow = [\"MPL-2.0\"] }}]\n"
        );
        assert!(
            policy_violations(&inline, false)[0]
                .contains("must be `[[licenses.exceptions]]` tables")
        );

        let wider = "[licenses]\nallow = [\"MIT\", \"MPL-2.0\"]\n";
        assert!(policy_violations(wider, false)[0].contains("must be exactly the 09 §7 list"));

        let ignores = format!(
            "[advisories]\nignore = [\"RUSTSEC-2021-0003\", {{ id = \"RUSTSEC-2020-0001\", reason = \"r\" }}]\n[licenses]\n{ALLOW}"
        );
        let prover = policy_violations(&ignores, false);
        assert_eq!(prover.len(), 1, "{prover:#?}");
        assert!(prover[0].contains("RUSTSEC-2021-0003 needs `reason"));
        assert_eq!(
            policy_violations(&ignores, true).len(),
            2,
            "the verifier ignores nothing"
        );
    }
}
