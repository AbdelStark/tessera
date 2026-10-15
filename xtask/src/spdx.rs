// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SPDX headers (09 §7): every Rust, Python and CUDA source file starts with the license
//! identifier `Apache-2.0 OR MIT`. A Python file may put a shebang line first.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The SPDX expression every source file declares.
pub(crate) const LICENSE: &str = "Apache-2.0 OR MIT";

/// The header line a file with this extension must carry, or `None` if the file is not checked.
fn expected_header(path: &Path) -> Option<String> {
    let comment = match path.extension()?.to_str()? {
        "rs" | "cu" | "cuh" => "//",
        "py" => "#",
        _ => return None,
    };
    Some(format!("{comment} SPDX-License-Identifier: {LICENSE}"))
}

/// Checks the files of `files` (relative to `repo`) and returns one message per violation.
pub(crate) fn check(repo: &Path, files: &[PathBuf]) -> io::Result<Vec<String>> {
    let mut violations = Vec::new();
    for rel in files {
        let Some(header) = expected_header(rel) else {
            continue;
        };
        let text = fs::read_to_string(repo.join(rel))?;
        let mut lines = text.lines();
        let mut first = lines.next();
        let is_python = rel.extension().is_some_and(|ext| ext == "py");
        if is_python && first.is_some_and(|line| line.starts_with("#!")) {
            first = lines.next();
        }
        if first.map(str::trim_end) != Some(header.as_str()) {
            violations.push(format!("{}: must start with `{header}`", rel.display()));
        }
    }
    Ok(violations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Fixture;

    #[test]
    fn files_without_the_header_are_reported() {
        let fixture = Fixture::new("spdx");
        let files = [
            (
                "ok.rs",
                format!("// SPDX-License-Identifier: {LICENSE}\n\nfn f() {{}}\n"),
            ),
            (
                "script.py",
                format!("#!/usr/bin/env python3\n# SPDX-License-Identifier: {LICENSE}\n"),
            ),
            (
                "kernel.cu",
                format!("// SPDX-License-Identifier: {LICENSE}\r\n"),
            ),
            ("notes.md", "no header needed\n".to_owned()),
            ("missing.rs", "fn f() {}\n".to_owned()),
            (
                "late.rs",
                format!("\n// SPDX-License-Identifier: {LICENSE}\n"),
            ),
            (
                "mit_only.cu",
                "// SPDX-License-Identifier: MIT\n".to_owned(),
            ),
            ("missing.py", "print('x')\n".to_owned()),
            ("empty.cuh", String::new()),
        ];
        let paths: Vec<PathBuf> = files
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
                "missing.rs",
                "late.rs",
                "mit_only.cu",
                "missing.py",
                "empty.cuh"
            ]
        );
    }
}
