// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Merge-conflict markers (07 §9): no file of the repository holds a line that opens a conflict,
//! the base's part of one (diff3), or its end. Markdown reads a marker as text, so a rebase that
//! leaves one in a document passes every other check. One left in the artifact catalog of 03 §5
//! once reached CI.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The line prefixes Git writes around a conflict. The `=======` between its two sides is not
/// one: a Markdown heading may be underlined with it, and a conflict always has a start and an
/// end.
const MARKERS: [&str; 3] = ["<<<<<<< ", "||||||| ", ">>>>>>> "];

/// Checks the files of `files` (relative to `repo`) and returns one message per marked line.
/// Files that are not UTF-8 text are not read as lines.
pub(crate) fn check(repo: &Path, files: &[PathBuf]) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    for rel in files {
        let bytes = fs::read(repo.join(rel))?;
        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        for (number, line) in text.lines().enumerate() {
            if MARKERS.iter().any(|marker| line.starts_with(marker)) {
                violations.push(format!(
                    "{}:{}: a merge-conflict marker",
                    rel.display(),
                    number + 1
                ));
            }
        }
    }
    Ok(violations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Fixture;

    #[test]
    fn every_marker_line_is_reported() {
        let fixture = Fixture::new("conflicts");
        // Built at run time, so that this file holds no marker itself.
        let start = format!("{}{}", "<<<<<<", "< HEAD");
        let base = format!("{}{}", "||||||", "| base");
        let end = format!("{}{}", ">>>>>>", "> a774be9 (spec: …)");
        let files = [
            ("clean.md", "# Title\n\nText.\n".to_owned()),
            ("underlined.md", "Title\n=======\n\nText.\n".to_owned()),
            (
                "conflict.md",
                format!("| a |\n{start}\n| b |\n=======\n| c |\n{end}\n"),
            ),
            (
                "diff3.rs",
                format!("{start}\nx\n{base}\ny\n=======\nz\n{end}\n"),
            ),
            ("indented.md", format!("  {start}\n")),
        ];
        let mut paths: Vec<PathBuf> = files
            .iter()
            .map(|(name, text)| {
                fixture.write(name, text);
                PathBuf::from(name)
            })
            .collect();
        // A binary file is skipped, even one whose bytes would read as a marker.
        let binary = fixture.path().join("model.d");
        fs::write(&binary, [&[0xff, 0xfe][..], start.as_bytes()].concat()).expect("write");
        paths.push(PathBuf::from("model.d"));
        let violations = check(fixture.path(), &paths).expect("fixture is readable");
        assert_eq!(
            violations,
            [
                "conflict.md:2: a merge-conflict marker",
                "conflict.md:6: a merge-conflict marker",
                "diff3.rs:1: a merge-conflict marker",
                "diff3.rs:3: a merge-conflict marker",
                "diff3.rs:7: a merge-conflict marker",
            ]
        );
    }
}
