// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Locating the repository and listing its files.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The repository root: the parent of the `xtask` crate.
pub(crate) fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default()
}

/// Files that are tracked, or untracked but not ignored, relative to `repo`, sorted.
///
/// Untracked files are included so that a new file is checked before it is committed.
pub(crate) fn files(repo: &Path) -> io::Result<Vec<PathBuf>> {
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(repo)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let mut files: Vec<PathBuf> = output
        .stdout
        .split(|&b| b == 0)
        .filter(|name| !name.is_empty())
        .map(|name| PathBuf::from(String::from_utf8_lossy(name).into_owned()))
        // The index can list files deleted from the working tree.
        .filter(|rel| repo.join(rel).is_file())
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}
