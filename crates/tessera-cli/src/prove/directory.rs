// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Producer-side RFC-0014 directory exchange, using only the producer codec.
use super::usage;
use crate::{Input, io_error};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};
use tessera_codec::{
    Decode, Encode, envelope,
    error::{ErrorCode, TesseraError},
    schema::{self, Schema},
};
use tessera_prover::{ProofBatch, Prover};

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
static NEXT: AtomicU64 = AtomicU64::new(0);

pub(super) fn publish(path: &Path, bytes: &[u8]) -> Result<(), TesseraError> {
    atomic(path, bytes, &NEXT)?;
    crate::manifest::output_envelope(path, bytes);
    Ok(())
}
/// RFC-0003 envelope encoding uses one cleared bounded buffer while the message lives.
/// Keep the directory protocol's exclusive 0600, sync, and no-clobber publication.
pub(super) fn publish_envelope<T: Encode + ?Sized>(
    path: &Path,
    schema: &'static Schema,
    value: &T,
) -> Result<(), TesseraError> {
    atomic_with(path, &NEXT, |file| {
        envelope::write(file, schema, value).map(|_| ())
    })?;
    crate::manifest::output_file(path, schema);
    Ok(())
}
fn atomic(path: &Path, bytes: &[u8], next: &AtomicU64) -> Result<(), TesseraError> {
    atomic_with(path, next, |file| file.write_all(bytes))
}
fn atomic_with(
    path: &Path,
    next: &AtomicU64,
    contents: impl FnOnce(&mut File) -> std::io::Result<()>,
) -> Result<(), TesseraError> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    for _ in 0..64 {
        let tmp = parent.join(format!(
            ".tessera-prover-{}-{}.tmp",
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
        let mut file = match options.open(&tmp) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(TesseraError::new(ErrorCode::ResIo)),
        };
        let guard = Temporary(tmp);
        contents(&mut file)
            .and_then(|_| file.sync_all())
            .map_err(io_error("cannot write protocol output".into()))?;
        drop(file);
        fs::hard_link(&guard.0, path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                usage("output already exists; use a fresh output path/session directory")
            } else {
                TesseraError::new(ErrorCode::ResIo)
            }
        })?;
        return Ok(());
    }
    Err(TesseraError::new(ErrorCode::ResIo))
}
fn receive<T: Decode>(
    dir: &Path,
    name: &str,
    schema: &'static Schema,
    timeout: Duration,
) -> Result<T, TesseraError> {
    let path = dir.join(name);
    let start = Instant::now();
    loop {
        if start.elapsed() >= timeout {
            return Err(TesseraError::new(ErrorCode::ProtoTimeout));
        }
        match fs::symlink_metadata(&path) {
            Ok(_) => return Input::new(&path, schema).open(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                std::thread::sleep(Duration::from_millis(5))
            }
            Err(_) => return Err(TesseraError::new(ErrorCode::ResIo)),
        }
    }
}
fn send<T: Encode>(
    dir: &Path,
    name: &str,
    schema: &'static Schema,
    message: &T,
) -> Result<(), TesseraError> {
    publish_envelope(&dir.join(name), schema, message)
}
pub(super) fn run(
    prover: &mut Prover<'_>,
    batch: &ProofBatch,
    dir: &Path,
    timeout: Duration,
) -> Result<(), TesseraError> {
    if !fs::metadata(dir)
        .map_err(io_error("cannot open session directory".into()))?
        .is_dir()
    {
        return Err(TesseraError::new(ErrorCode::ResIo));
    }
    // Each role is one-shot even when the peer has already started.
    publish(&dir.join(".tessera-prover"), b"")?;
    let commit = receive(dir, "00-SeedCommit.tce", &schema::SEED_COMMIT, timeout)?;
    let (first, share) = prover.start(batch, &commit).map_err(super::prover_error)?;
    send(dir, "01-Round1.tce", &schema::ROUND1, &first)?;
    send(
        dir,
        "01-ProverSeedShare.tce",
        &schema::PROVER_SEED_SHARE,
        &share,
    )?;
    let reveal = receive(dir, "02-SeedReveal.tce", &schema::SEED_REVEAL, timeout)?;
    let second = prover.on_seed(&reveal).map_err(super::prover_error)?;
    // The seed both transcripts absorb is public; the shares behind it were revealed (RFC-0014).
    crate::manifest::seed(
        "coin_toss",
        &tessera_protocol::coin_toss(&reveal.share, &share.share),
    );
    send(
        dir,
        "03-Round2Response.tce",
        &schema::ROUND2_RESPONSE,
        &second,
    )?;
    let queries = receive(
        dir,
        "04-ColumnQueries.tce",
        &schema::COLUMN_QUERIES,
        timeout,
    )?;
    let third = prover.on_queries(&queries).map_err(super::prover_error)?;
    send(
        dir,
        "05-Round3Response.tce",
        &schema::ROUND3_RESPONSE,
        &third,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_publication_is_exact_bounded_and_never_clobbers() {
        use allocation_counter::measure;
        let dir =
            std::env::temp_dir().join(format!("tessera-stream-publication-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        for size in [1 << 20, 8 << 20] {
            let path = dir.join(format!("packet-{size}"));
            let payload = vec![0xA5u8; size]; // Public synthetic bytes, not a proof.
            let expected = envelope::seal(&schema::NI_PROOF, &payload);
            let mut outcome = None;
            let allocations = measure(|| {
                outcome = Some(publish_envelope(&path, &schema::NI_PROOF, &payload));
            });
            outcome.unwrap().unwrap();
            assert!(allocations.bytes_max <= 96 << 10, "{allocations:?}");
            assert_eq!(fs::read(&path).unwrap(), expected);
            assert_eq!(
                publish_envelope(&path, &schema::NI_PROOF, &0u64)
                    .unwrap_err()
                    .code,
                ErrorCode::Usage
            );
            assert_eq!(fs::read(&path).unwrap(), expected);
            #[cfg(unix)]
            {
                use std::os::unix::fs::{PermissionsExt, symlink};
                assert_eq!(
                    fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                    0o600
                );
                let alias = dir.join(format!("alias-{size}"));
                symlink(&path, &alias).unwrap();
                assert_eq!(
                    publish_envelope(&alias, &schema::NI_PROOF, &0u64)
                        .unwrap_err()
                        .code,
                    ErrorCode::Usage
                );
                assert_eq!(fs::read(&path).unwrap(), expected);
                fs::remove_file(alias).unwrap();
            }
        }
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        // Encoding or I/O failure must leave neither a partial output nor a temporary.
        let path = dir.join("failed");
        assert_eq!(
            atomic_with(&path, &AtomicU64::new(0), |file| {
                file.write_all(b"partial")?;
                Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "injected",
                ))
            })
            .unwrap_err()
            .code,
            ErrorCode::ResIo
        );
        assert!(!path.exists());
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        // The streaming codec checks that both Encode passes agree before publishing.
        struct Changing(std::cell::Cell<bool>);
        impl Encode for Changing {
            fn encode_to<S: tessera_codec::Sink + ?Sized>(&self, sink: &mut S) {
                if self.0.replace(true) {
                    sink.put(b"x");
                }
            }
        }
        assert_eq!(
            publish_envelope(
                &path,
                &schema::NI_PROOF,
                &Changing(std::cell::Cell::new(false))
            )
            .unwrap_err()
            .code,
            ErrorCode::ResIo
        );
        assert!(!path.exists());
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn publishing_preserves_outputs_and_receive_checks_deadlines_and_envelopes() {
        let dir =
            std::env::temp_dir().join(format!("tessera-prover-directory-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("packet");
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            fs::write(dir.join("canary"), b"unowned").unwrap();
            for i in 0..64 {
                symlink(
                    dir.join("canary"),
                    dir.join(format!(".tessera-prover-{}-{i}.tmp", std::process::id())),
                )
                .unwrap();
            }
            assert_eq!(
                atomic(&path, b"overwrite", &AtomicU64::new(0))
                    .unwrap_err()
                    .code,
                ErrorCode::ResIo
            );
            assert_eq!(fs::read(dir.join("canary")).unwrap(), b"unowned");
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 65);
            for i in 0..64 {
                fs::remove_file(
                    dir.join(format!(".tessera-prover-{}-{i}.tmp", std::process::id())),
                )
                .unwrap();
            }
            fs::remove_file(dir.join("canary")).unwrap();
        }
        publish(&path, b"original").unwrap();
        assert_eq!(
            publish(&path, b"replacement").unwrap_err().code,
            ErrorCode::Usage
        );
        assert_eq!(fs::read(&path).unwrap(), b"original");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        assert_eq!(
            receive::<tessera_protocol::SeedCommit>(
                &dir,
                "packet",
                &schema::SEED_COMMIT,
                Duration::ZERO
            )
            .unwrap_err()
            .code,
            ErrorCode::ProtoTimeout
        );
        assert_eq!(
            receive::<tessera_protocol::SeedCommit>(
                &dir,
                "packet",
                &schema::SEED_COMMIT,
                Duration::from_secs(1)
            )
            .unwrap_err()
            .code,
            ErrorCode::CodecMagic
        );
        fs::create_dir(dir.join("not-a-file")).unwrap();
        assert_eq!(
            receive::<tessera_protocol::SeedCommit>(
                &dir,
                "not-a-file",
                &schema::SEED_COMMIT,
                Duration::from_secs(1)
            )
            .unwrap_err()
            .code,
            ErrorCode::ResIo
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::{PermissionsExt, symlink};
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            symlink(&path, dir.join("alias")).unwrap();
            assert_eq!(
                publish(&dir.join("alias"), b"replacement")
                    .unwrap_err()
                    .code,
                ErrorCode::Usage
            );
            assert_eq!(fs::read(&path).unwrap(), b"original");
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
