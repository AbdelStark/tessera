// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tv_core::verdict::ErrorCode;

pub(crate) struct Inputs {
    pub(crate) remaining: u64,
    pub(crate) max_artifact: u64,
}
impl Inputs {
    pub(crate) fn read(&mut self, path: &Path) -> Result<Vec<u8>, ErrorCode> {
        // shortcut: regular local files and cooperative deadlines; hard I/O deadlines are #182.
        if !fs::metadata(path).map_err(|_| ErrorCode::Io)?.is_file() {
            return Err(ErrorCode::Io);
        }
        let (mut file, len) = open_input(path)?;
        self.read_from(&mut file, len)
    }

    fn read_from(
        &mut self,
        reader: &mut impl Read,
        initial_len: u64,
    ) -> Result<Vec<u8>, ErrorCode> {
        let limit = self.max_artifact.min(self.remaining);
        let exceeded = if self.remaining < self.max_artifact {
            ErrorCode::Memory
        } else {
            ErrorCode::CodecLimit
        };
        let out = read_bounded(reader, initial_len, limit, exceeded)?;
        self.remaining = self
            .remaining
            .checked_sub(out.len() as u64)
            .ok_or(ErrorCode::Memory)?;
        Ok(out)
    }
}

// O_NONBLOCK: Linux asm-generic/fcntl.h (also arm/arm64), Apple sys/fcntl.h.
#[cfg(unix)]
const INPUT_NONBLOCK: i32 = if cfg!(target_vendor = "apple") {
    0x0004
} else if cfg!(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "riscv64",
        target_arch = "aarch64",
        target_arch = "arm"
    )
)) {
    0o4_000
} else {
    // shortcut: unnamed Unix flags keep preflight alone; qualify the host before relying on it.
    0
};

// The path can change after preflight; validate the handle actually opened.
fn open_input(path: &Path) -> Result<(File, u64), ErrorCode> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(INPUT_NONBLOCK);
    }
    let file = options.open(path).map_err(|_| ErrorCode::Io)?;
    let meta = file.metadata().map_err(|_| ErrorCode::Io)?;
    if !meta.is_file() {
        return Err(ErrorCode::Io);
    }
    Ok((file, meta.len()))
}

fn read_bounded(
    reader: &mut impl Read,
    initial_len: u64,
    limit: u64,
    exceeded: ErrorCode,
) -> Result<Vec<u8>, ErrorCode> {
    if initial_len > limit {
        return Err(exceeded);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(usize::try_from(initial_len).map_err(|_| ErrorCode::Memory)?)
        .map_err(|_| ErrorCode::Memory)?;
    let mut scratch = [0; 8192];
    loop {
        let n = reader.read(&mut scratch).map_err(|_| ErrorCode::Io)?;
        if n == 0 {
            break;
        }
        let len = out.len().checked_add(n).ok_or(ErrorCode::Memory)?;
        if u64::try_from(len).map_err(|_| ErrorCode::Memory)? > limit {
            return Err(exceeded);
        }
        out.try_reserve(n).map_err(|_| ErrorCode::Memory)?;
        out.extend_from_slice(&scratch[..n]);
    }
    Ok(out)
}

/// The file `path` names: its canonical path when it exists, else its directory's joined with
/// its name. `ERR-RES-IO` when neither resolves.
fn normalized(path: &Path) -> Result<PathBuf, ErrorCode> {
    fs::canonicalize(path)
        .or_else(|_| {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            Ok::<_, std::io::Error>(
                fs::canonicalize(parent)?
                    .join(path.file_name().ok_or(std::io::ErrorKind::InvalidInput)?),
            )
        })
        .map_err(|_| ErrorCode::Io)
}

/// Whether two normalized paths may name one file: the same path, or names in one directory
/// that differ only in ASCII case, which a case-insensitive filesystem (APFS, HFS+, NTFS by
/// default) resolves to one entry. An output that does not exist yet keeps its name as given, so
/// equality alone cannot see that alias.
fn may_alias(a: &Path, b: &Path) -> bool {
    a == b
        || (a.parent() == b.parent()
            && match (a.file_name(), b.file_name()) {
                (Some(x), Some(y)) => x
                    .as_encoded_bytes()
                    .eq_ignore_ascii_case(y.as_encoded_bytes()),
                _ => false,
            })
}

/// `out` names no input: `ERR-USAGE` when it resolves to one. Every command's output takes this
/// check, unchanged since before #446; the case rule is the run manifest's alone.
pub(crate) fn output_distinct(out: Option<&str>, inputs: &[&str]) -> Result<(), ErrorCode> {
    if let Some(out) = out {
        let normalized = normalized(Path::new(out))?;
        for input in inputs {
            if fs::canonicalize(input).is_ok_and(|p| p == normalized) {
                return Err(ErrorCode::Usage);
            }
        }
    }
    Ok(())
}

/// The run manifest's `path` names no input and not the report at `out`, in any ASCII case of
/// its name in one directory (05 §6 'The verifier's manifest'): `ERR-USAGE` otherwise. Before
/// verification neither output exists, so this is what can be seen then; [`same_file`] catches
/// the aliases only the filesystem knows once the report exists.
pub(crate) fn manifest_distinct(
    path: &str,
    inputs: &[&str],
    out: Option<&str>,
) -> Result<(), ErrorCode> {
    let manifest = normalized(Path::new(path))?;
    for input in inputs {
        if fs::canonicalize(input).is_ok_and(|p| may_alias(&p, &manifest)) {
            return Err(ErrorCode::Usage);
        }
    }
    if let Some(out) = out
        && may_alias(&normalized(Path::new(out))?, &manifest)
    {
        return Err(ErrorCode::Usage);
    }
    Ok(())
}

/// Whether `a` now resolves to the file `b` names: their canonical paths are equal, which also
/// sees a link and the filesystem's own case folding and Unicode normalization, or, on Unix,
/// they are one inode of one device. A path that does not resolve is no file.
pub(crate) fn same_file(a: &Path, b: &Path) -> bool {
    if let (Ok(x), Ok(y)) = (fs::canonicalize(a), fs::canonicalize(b))
        && x == y
    {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(x), Ok(y)) = (fs::metadata(a), fs::metadata(b)) {
            return x.dev() == y.dev() && x.ino() == y.ino();
        }
    }
    false
}

struct Temporary {
    path: PathBuf,
    armed: bool,
}
impl Drop for Temporary {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}
static NEXT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn write(path: &Path, bytes: &[u8]) -> Result<(), ErrorCode> {
    atomic(path, bytes, true, &NEXT)
}

pub(crate) fn publish(path: &Path, bytes: &[u8]) -> Result<(), ErrorCode> {
    atomic(path, bytes, false, &NEXT)
}

/// Syncs the directory that holds `path`, so that a rename into it survives a power loss: on
/// Unix `fsync` of the directory; other hosts give `std` no handle on a directory.
pub(crate) fn sync_parent(path: &Path) -> Result<(), ErrorCode> {
    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        File::open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(|_| ErrorCode::Io)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

fn atomic(path: &Path, bytes: &[u8], replace: bool, next: &AtomicU64) -> Result<(), ErrorCode> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    for _ in 0..64 {
        let temp = parent.join(format!(
            ".tessera-verdict-{}-{}.tmp",
            std::process::id(),
            next.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&temp) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(ErrorCode::Io),
        };
        let mut guard = Temporary {
            path: temp,
            armed: true,
        };
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|_| ErrorCode::Io)?;
        drop(file);
        if replace {
            fs::rename(&guard.path, path).map_err(|_| ErrorCode::Io)?;
            guard.armed = false;
        } else {
            fs::hard_link(&guard.path, path).map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    ErrorCode::Usage
                } else {
                    ErrorCode::Io
                }
            })?;
        }
        return Ok(());
    }
    Err(ErrorCode::Io)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(
        target_vendor = "apple",
        all(
            target_os = "linux",
            any(
                target_arch = "x86_64",
                target_arch = "x86",
                target_arch = "riscv64",
                target_arch = "aarch64",
                target_arch = "arm"
            )
        )
    ))]
    #[test]
    fn input_open_refuses_a_fifo_swapped_after_preflight() {
        use std::process::{Command, Stdio};
        const CHILD_PATH: &str = "TESSERA_TEST_INPUT_FIFO";
        if let Some(path) = std::env::var_os(CHILD_PATH) {
            assert_eq!(open_input(Path::new(&path)).err(), Some(ErrorCode::Io));
            fs::write(Path::new(&path).with_extension("done"), b"Io").unwrap();
            return;
        }
        let dir = std::env::temp_dir().join(format!("tessera-input-swap-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("input");
        fs::write(&path, b"regular input").unwrap();
        assert!(fs::metadata(&path).unwrap().is_file());
        let mut input = Inputs {
            remaining: 13,
            max_artifact: 13,
        };
        assert_eq!(input.read(&path).unwrap(), b"regular input");
        assert_eq!(input.remaining, 0);
        // Replace the path between the preflight check and its shared open operation.
        fs::rename(&path, dir.join("original")).unwrap();
        assert!(
            Command::new("mkfifo")
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "files::tests::input_open_refuses_a_fifo_swapped_after_preflight",
            ])
            .env(CHILD_PATH, &path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                child.wait().unwrap();
                break None;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert_eq!(fs::read(dir.join("original")).unwrap(), b"regular input");
        let marker = fs::read(path.with_extension("done")).ok();
        fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            marker.as_deref(),
            Some(b"Io".as_slice()),
            "child did not reach its assertion"
        );
        assert!(
            status.is_some_and(|s| s.success()),
            "input open blocked on a swapped FIFO"
        );
    }

    /// Whether names that differ only in case are one file in the temporary directory.
    fn folds_case(dir: &Path) -> bool {
        fs::write(dir.join("case-probe"), b"").unwrap();
        fs::metadata(dir.join("CASE-PROBE")).is_ok()
    }

    /// The case rule is the run manifest's alone: `--out`, and every command's output with it
    /// (`verify`, `live`, `audit report`, `audit reveal`), is refused only when it resolves to an
    /// input, as before #446; the manifest's path is refused in any ASCII case of an input's
    /// name or of the report's.
    #[test]
    fn the_case_rule_binds_the_manifest_alone() {
        let dir = std::env::temp_dir().join(format!("tessera-case-rule-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("input.bin"), b"input").unwrap();
        let path = |name: &str| dir.join(name).display().to_string();
        let input = path("input.bin");
        let folds = folds_case(&dir);
        let out = output_distinct(Some(&path("INPUT.BIN")), &[&input]);
        let exact = output_distinct(Some(&input), &[&input]);
        let manifest = manifest_distinct(&path("INPUT.BIN"), &[&input], None);
        let report = manifest_distinct(&path("r.json"), &[&input], Some(&path("R.json")));
        let apart = manifest_distinct(&path("m.json"), &[&input], Some(&path("r.json")));
        fs::remove_dir_all(&dir).unwrap();
        // A case variant of an input is that input only where the filesystem folds case.
        assert_eq!(out, if folds { Err(ErrorCode::Usage) } else { Ok(()) });
        assert_eq!(exact, Err(ErrorCode::Usage));
        assert_eq!(manifest, Err(ErrorCode::Usage));
        assert_eq!(report, Err(ErrorCode::Usage));
        assert_eq!(apart, Ok(()));
    }

    #[test]
    fn names_that_differ_only_in_ascii_case_in_one_directory_may_alias() {
        for (a, b, alias) in [
            ("/d/r.json", "/d/r.json", true),
            ("/d/r.json", "/d/R.JSON", true),
            ("/d/r.json", "/e/R.json", false),
            ("/d/r.json", "/d/r.jsonx", false),
            // Only ASCII letters fold: `é` and `É` are other bytes.
            ("/d/\u{e9}.json", "/d/\u{c9}.json", false),
            ("/", "/", true),
        ] {
            assert_eq!(may_alias(Path::new(a), Path::new(b)), alias, "{a} {b}");
        }
    }
    #[test]
    fn empty_read_terminates_without_retrying_eof() {
        struct Eof(bool);
        impl Read for Eof {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                assert!(!self.0, "retried EOF");
                self.0 = true;
                Ok(0)
            }
        }
        assert_eq!(
            read_bounded(&mut Eof(false), 0, 0, ErrorCode::CodecLimit),
            Ok(vec![])
        );
    }
    #[test]
    fn successive_inputs_share_one_budget() {
        let path =
            std::env::temp_dir().join(format!("tessera-input-budget-{}", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&[1, 2]).unwrap();
        drop(file);
        let mut input = Inputs {
            remaining: 4,
            max_artifact: 2,
        };
        assert_eq!(input.read(&path), Ok(vec![1, 2]));
        assert_eq!(input.remaining, 2);
        assert_eq!(input.read(&path), Ok(vec![1, 2]));
        assert_eq!(input.remaining, 0);
        let denied = input.read(&path);
        fs::remove_file(&path).unwrap();
        assert_eq!(denied, Err(ErrorCode::Memory));
    }
    #[cfg(unix)]
    #[test]
    fn report_temp_collisions_preserve_unowned_files_and_stop() {
        let dir =
            std::env::temp_dir().join(format!("tessera-report-collisions-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let canary = dir.join("canary");
        fs::write(&canary, b"keep this file").unwrap();
        for i in 0..64 {
            std::os::unix::fs::symlink(
                &canary,
                dir.join(format!(".tessera-verdict-{}-{i}.tmp", std::process::id())),
            )
            .unwrap();
        }
        let result = atomic(
            &dir.join("verdict"),
            b"overwrite attempt",
            true,
            &AtomicU64::new(0),
        );
        let contents = fs::read(&canary).unwrap();
        let count = fs::read_dir(&dir).unwrap().count();
        fs::remove_dir_all(&dir).unwrap();
        assert_eq!(result, Err(ErrorCode::Io));
        assert_eq!(contents, b"keep this file");
        assert_eq!(count, 65);
    }
    #[test]
    fn input_limits_cover_metadata_growth_and_read_errors() {
        for initial in [0, 2] {
            assert_eq!(
                read_bounded(&mut &[1, 2][..], initial, 2, ErrorCode::CodecLimit),
                Ok(vec![1, 2])
            );
            assert_eq!(
                read_bounded(&mut &[1, 2, 3][..], initial, 2, ErrorCode::CodecLimit),
                Err(ErrorCode::CodecLimit)
            );
        }
        assert_eq!(
            read_bounded(&mut &[][..], 3, 2, ErrorCode::Memory),
            Err(ErrorCode::Memory)
        );
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::Other.into())
            }
        }
        assert_eq!(
            read_bounded(&mut Broken, 0, 2, ErrorCode::CodecLimit),
            Err(ErrorCode::Io)
        );
        let mut input = Inputs {
            remaining: 1,
            max_artifact: 2,
        };
        // The shared memory budget must win before any allocation or failing read.
        assert_eq!(input.read_from(&mut Broken, 2), Err(ErrorCode::Memory));
    }
}
