// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask check-pr`: the pull-request policies of docs/spec/09-release-and-versioning.md
//! that depend on the change rather than on the tree.
//!
//! - **Changelog (09 §5).** A change to a release train adds an entry to that train's
//!   changelog: `verifier/` needs `verifier/CHANGELOG.md`; `crates/*/src` and
//!   `python/tessera_quant/` need `CHANGELOG.md`; `spec/`, which both trains read, needs either.
//!   The `no-changelog` label waives the rule.
//! - **Developer Certificate of Origin (09 §6).** Every commit that is not a merge carries a
//!   `Signed-off-by:` line naming its author's email address.

use std::io;
use std::path::Path;
use std::process::Command;

/// The label that waives the changelog rule.
pub(crate) const NO_CHANGELOG: &str = "no-changelog";

const PROVER_CHANGELOG: &str = "CHANGELOG.md";
const VERIFIER_CHANGELOG: &str = "verifier/CHANGELOG.md";

/// The changelog a changed path requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Train {
    Prover,
    Verifier,
    /// Shared data that both trains read (`spec/`): either changelog will do.
    Either,
}

fn required_changelog(path: &str) -> Option<Train> {
    if let Some(rest) = path.strip_prefix("verifier/") {
        return (rest != "CHANGELOG.md").then_some(Train::Verifier);
    }
    if path.starts_with("python/tessera_quant/") {
        return Some(Train::Prover);
    }
    if path.starts_with("spec/") {
        return Some(Train::Either);
    }
    let mut parts = path.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("crates"), Some(_), Some("src")) => Some(Train::Prover),
        _ => None,
    }
}

/// Changelog violations for a change touching `changed` (paths relative to the repository root)
/// on a pull request carrying `labels`.
pub(crate) fn changelog_violations(changed: &[String], labels: &[String]) -> Vec<String> {
    if labels.iter().any(|label| label == NO_CHANGELOG) {
        return Vec::new();
    }
    let touched = |file: &str| changed.iter().any(|path| path == file);
    let first = |train: Train| {
        changed
            .iter()
            .find(|p| required_changelog(p) == Some(train))
    };
    let waiver = format!("or label the pull request `{NO_CHANGELOG}`");
    let mut violations = Vec::new();
    if let Some(path) = first(Train::Prover).filter(|_| !touched(PROVER_CHANGELOG)) {
        violations.push(format!(
            "{path} changes the prover release train: add an entry under Unreleased in \
             {PROVER_CHANGELOG}, {waiver}"
        ));
    }
    if let Some(path) = first(Train::Verifier).filter(|_| !touched(VERIFIER_CHANGELOG)) {
        violations.push(format!(
            "{path} changes the verifier release train: add an entry under Unreleased in \
             {VERIFIER_CHANGELOG}, {waiver}"
        ));
    }
    if let Some(path) =
        first(Train::Either).filter(|_| !touched(PROVER_CHANGELOG) && !touched(VERIFIER_CHANGELOG))
    {
        violations.push(format!(
            "{path} changes shared specification data: add an entry under Unreleased in \
             {PROVER_CHANGELOG} or {VERIFIER_CHANGELOG}, {waiver}"
        ));
    }
    violations
}

/// A commit of the pull request.
#[derive(Debug)]
pub(crate) struct Commit {
    pub(crate) sha: String,
    pub(crate) author_email: String,
    pub(crate) message: String,
}

/// The email address of a `Signed-off-by: Name <email>` line, if the line is one.
fn sign_off_email(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("Signed-off-by:")?;
    let start = rest.rfind('<')?;
    let end = rest.rfind('>')?;
    (start < end).then(|| rest[start + 1..end].trim())
}

/// DCO violations: commits without a sign-off naming their author's email address.
pub(crate) fn dco_violations(commits: &[Commit]) -> Vec<String> {
    commits
        .iter()
        .filter(|c| {
            !c.message
                .lines()
                .filter_map(sign_off_email)
                .any(|email| email.eq_ignore_ascii_case(&c.author_email))
        })
        .map(|c| {
            let short = c.sha.get(..12).unwrap_or(&c.sha);
            let subject = c.message.lines().next().unwrap_or_default();
            format!(
                "{short} \"{subject}\": no `Signed-off-by:` line for its author <{}>; \
                 sign off with `git commit -s` (09 §6)",
                c.author_email
            )
        })
        .collect()
}

fn git(repo: &Path, args: &[&str]) -> io::Result<String> {
    let output = Command::new("git").args(args).current_dir(repo).output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    String::from_utf8(output.stdout).map_err(io::Error::other)
}

/// Paths changed between the merge base of `base` and `HEAD`, and `HEAD`.
pub(crate) fn changed_files(repo: &Path, base: &str) -> io::Result<Vec<String>> {
    let out = git(
        repo,
        &["diff", "--name-only", "-z", &format!("{base}...HEAD")],
    )?;
    Ok(out
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
        .collect())
}

/// The commits in `base..HEAD` that are not merges.
pub(crate) fn commits(repo: &Path, base: &str) -> io::Result<Vec<Commit>> {
    // Fields end in NUL, records in RS (0x1e); neither occurs in commit metadata.
    let out = git(
        repo,
        &[
            "log",
            "--no-merges",
            "--format=%H%x00%ae%x00%B%x1e",
            &format!("{base}..HEAD"),
        ],
    )?;
    Ok(out
        .split('\u{1e}')
        .filter_map(|record| {
            let mut fields = record.trim_start_matches('\n').splitn(3, '\0');
            let (sha, email, message) = (fields.next()?, fields.next()?, fields.next()?);
            (!sha.is_empty()).then(|| Commit {
                sha: sha.to_owned(),
                author_email: email.to_owned(),
                message: message.to_owned(),
            })
        })
        .collect())
}

/// Runs both checks for the change `base...HEAD` and reports on stdout.
pub(crate) fn run(repo: &Path, base: &str, labels: &[String]) -> io::Result<bool> {
    let changelog = changelog_violations(&changed_files(repo, base)?, labels);
    let dco = dco_violations(&commits(repo, base)?);
    let mut clean = true;
    for (name, violations) in [
        ("changelog entry (09 §5)", changelog),
        ("DCO sign-off (09 §6)", dco),
    ] {
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
    use crate::testing::Fixture;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn each_release_train_needs_its_own_changelog() {
        let cases: [(&[&str], &[&str], &[&str]); 11] = [
            // (changed files, labels, expected violation fragments)
            (
                &["crates/tessera-field/src/lib.rs"],
                &[],
                &["prover release train"],
            ),
            (
                &["crates/tessera-field/src/lib.rs", "CHANGELOG.md"],
                &[],
                &[],
            ),
            (&["crates/tessera-field/src/lib.rs"], &["no-changelog"], &[]),
            (
                &[
                    "crates/tessera-field/Cargo.toml",
                    "docs/spec/00-overview.md",
                    "README.md",
                ],
                &[],
                &[],
            ),
            (
                &["python/tessera_quant/tce.py"],
                &[],
                &["prover release train"],
            ),
            (
                &["verifier/crates/tv-core/src/lib.rs", "CHANGELOG.md"],
                &[],
                &["verifier release train"],
            ),
            (
                &["verifier/allowlist.toml", "verifier/CHANGELOG.md"],
                &[],
                &[],
            ),
            (&["verifier/CHANGELOG.md"], &[], &[]),
            (
                &["spec/vectors/field/goldilocks.json"],
                &[],
                &["shared specification data"],
            ),
            (
                &[
                    "spec/vectors/field/goldilocks.json",
                    "verifier/CHANGELOG.md",
                ],
                &[],
                &[],
            ),
            (
                &[
                    "crates/tessera-hash/src/lib.rs",
                    "verifier/crates/tv-hash/src/lib.rs",
                ],
                &["other-label"],
                &["prover release train", "verifier release train"],
            ),
        ];
        for (changed, labels, expected) in cases {
            let violations = changelog_violations(&strings(changed), &strings(labels));
            assert_eq!(
                violations.len(),
                expected.len(),
                "{changed:?} {labels:?}: {violations:#?}"
            );
            for (violation, fragment) in violations.iter().zip(expected) {
                assert!(violation.contains(fragment), "{violation} lacks {fragment}");
            }
        }
    }

    #[test]
    fn sign_off_must_name_the_author() {
        let commit = |email: &str, message: &str| Commit {
            sha: "0123456789abcdef".to_owned(),
            author_email: email.to_owned(),
            message: message.to_owned(),
        };
        let commits = [
            commit("a@example.com", "fix\n\nSigned-off-by: A <a@example.com>\n"),
            commit(
                "A@Example.com",
                "case\n\nSigned-off-by: A Person <a@example.COM>",
            ),
            commit(
                "a@example.com",
                "two\n\nSigned-off-by: B <b@example.com>\nSigned-off-by: A <a@example.com>",
            ),
            commit("a@example.com", "missing\n\nNo sign-off here."),
            commit(
                "a@example.com",
                "someone else\n\nSigned-off-by: B <b@example.com>",
            ),
            commit(
                "a@example.com",
                "mentioned\n\nas in Signed-off-by: A <a@example.com>",
            ),
        ];
        let violations = dco_violations(&commits);
        assert_eq!(violations.len(), 3, "{violations:#?}");
        assert!(violations[0].contains("\"missing\""));
        assert!(violations[1].contains("\"someone else\""));
        assert!(violations[2].contains("\"mentioned\""));
    }

    #[test]
    fn changes_and_commits_are_read_from_git() {
        let fixture = Fixture::new("pr-policy");
        let repo = fixture.path();
        let run_git = |args: &[&str]| {
            let mut all = vec![
                "-c",
                "user.name=Author",
                "-c",
                "user.email=author@example.com",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "init.defaultBranch=main",
            ];
            all.extend_from_slice(args);
            git(repo, &all).expect("git succeeds");
        };
        run_git(&["init", "-q"]);
        fixture.write("README.md", "base\n");
        run_git(&["add", "."]);
        run_git(&["commit", "-q", "-m", "base"]);
        let base = git(repo, &["rev-parse", "HEAD"])
            .expect("rev-parse")
            .trim()
            .to_owned();
        fixture.write("crates/tessera-x/src/lib.rs", "// change\n");
        run_git(&["add", "."]);
        run_git(&["commit", "-q", "-s", "-m", "signed change"]);
        fixture.write("crates/tessera-x/src/more.rs", "// more\n");
        run_git(&["add", "."]);
        run_git(&["commit", "-q", "-m", "unsigned change"]);

        let changed = changed_files(repo, &base).expect("diff");
        assert_eq!(
            changed,
            [
                "crates/tessera-x/src/lib.rs",
                "crates/tessera-x/src/more.rs"
            ]
        );
        let commits = commits(repo, &base).expect("log");
        assert_eq!(commits.len(), 2);
        let violations = dco_violations(&commits);
        assert_eq!(violations.len(), 1, "{violations:#?}");
        assert!(violations[0].contains("\"unsigned change\""));
        assert_eq!(changelog_violations(&changed, &[]).len(), 1);
    }
}
