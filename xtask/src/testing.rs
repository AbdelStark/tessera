// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Temporary fixture directories for policy tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A directory under the system temporary directory, removed on drop.
pub(crate) struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    /// Creates an empty fixture directory unique to this process and call.
    pub(crate) fn new(tag: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("tessera-xtask-{tag}-{}-{n}", std::process::id()));
        // A stale directory can only come from a crashed earlier run with the same pid.
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create fixture directory");
        Self { dir }
    }

    /// The fixture's root directory.
    pub(crate) fn path(&self) -> &Path {
        &self.dir
    }

    /// Writes `text` to `rel`, creating parent directories.
    pub(crate) fn write(&self, rel: &str, text: &str) {
        let path = self.dir.join(rel);
        fs::create_dir_all(path.parent().expect("fixture paths have a parent")).expect("mkdir");
        fs::write(path, text).expect("write fixture file");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
