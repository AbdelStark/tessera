// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Locating the repository and listing its files.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use toml::{Table, Value as Toml};

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
        // The index can list files deleted from the working tree; symlinks are skipped so
        // each regular file is checked once at its canonical path.
        .filter(|rel| fs::symlink_metadata(repo.join(rel)).is_ok_and(|meta| meta.is_file()))
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

/// The `[toolchain]` table of a `rust-toolchain.toml`, or `None` if the file is absent or has none.
pub(crate) fn toolchain_table(path: &Path) -> io::Result<Option<Toml>> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(text
            .parse::<Table>()
            .ok()
            .and_then(|mut t| t.remove("toolchain"))),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// The `channel` a `rust-toolchain.toml` pins, if any.
pub(crate) fn pinned_channel(path: &Path) -> Option<String> {
    toolchain_table(path)
        .ok()
        .flatten()?
        .get("channel")?
        .as_str()
        .map(str::to_owned)
}

/// Copies the directory tree `from` into `to`, creating `to`. Symlinks are not followed.
pub(crate) fn copy_dir(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}
