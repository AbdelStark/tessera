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
        let mut file = File::open(path).map_err(|_| ErrorCode::Io)?;
        let meta = file.metadata().map_err(|_| ErrorCode::Io)?;
        if !meta.is_file() {
            return Err(ErrorCode::Io);
        }
        self.read_from(&mut file, meta.len())
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

pub(crate) fn output_distinct(out: Option<&str>, inputs: &[&str]) -> Result<(), ErrorCode> {
    if let Some(out) = out {
        let out = Path::new(out);
        let normalized = fs::canonicalize(out)
            .or_else(|_| {
                let parent = out
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                Ok::<_, std::io::Error>(
                    fs::canonicalize(parent)?
                        .join(out.file_name().ok_or(std::io::ErrorKind::InvalidInput)?),
                )
            })
            .map_err(|_| ErrorCode::Io)?;
        for input in inputs {
            if fs::canonicalize(input).is_ok_and(|p| p == normalized) {
                return Err(ErrorCode::Usage);
            }
        }
    }
    Ok(())
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
pub(crate) fn write(path: &Path, bytes: &[u8]) -> Result<(), ErrorCode> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    for _ in 0..64 {
        let temp = parent.join(format!(
            ".tessera-verdict-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
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
        fs::rename(&guard.path, path).map_err(|_| ErrorCode::Io)?;
        guard.armed = false;
        return Ok(());
    }
    Err(ErrorCode::Io)
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let result = write(&dir.join("verdict"), b"overwrite attempt");
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
