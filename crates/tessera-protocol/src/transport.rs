// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0014 Live transports. Neither framing nor filenames enter the transcript.
//!
//! A transport belongs to one session. Any failed operation makes it unusable; drop
//! the associated prover session too, so INV-PROTO-1 discards its committed randomness.
//! Typed receives use the existing canonical decoder, with caller-owned limits.
//! TLS is available with the opt-in `tls` feature; no network stack is needed for directories.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use tessera_codec::{
    DecodeLimits, Sink, envelope,
    error::{ErrorCode, TesseraError},
};

use crate::{Message, MessageKind};

#[cfg(feature = "tls")]
pub mod tls;

/// Per-message absolute time budgets (RFC-0014, OQ-0014-1). Not transcript parameters.
#[derive(Clone, Copy, Debug)]
pub struct Deadlines {
    /// Default: one hour for the witness/proof response.
    pub round2: Duration,
    /// Default: ten minutes for every other Live message.
    pub other: Duration,
}

impl Default for Deadlines {
    fn default() -> Self {
        Self {
            round2: Duration::from_secs(3600),
            other: Duration::from_secs(600),
        }
    }
}

impl Deadlines {
    /// Budget for one message; progress does not restart it.
    pub fn for_message(self, kind: MessageKind) -> Duration {
        if kind == MessageKind::Round2Response {
            self.round2
        } else {
            self.other
        }
    }
}

fn error(code: ErrorCode) -> TesseraError {
    TesseraError::new(code)
}

fn io_error(cause: io::Error) -> TesseraError {
    error(match cause.kind() {
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => ErrorCode::ProtoTimeout,
        _ => ErrorCode::ResIo,
    })
}

fn deadline(duration: Duration) -> Result<Instant, TesseraError> {
    Instant::now()
        .checked_add(duration)
        .ok_or_else(|| error(ErrorCode::Usage))
}

fn remaining(end: Instant) -> io::Result<Duration> {
    end.checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| io::Error::from(io::ErrorKind::TimedOut))
}

fn live(kind: MessageKind) -> Result<(), TesseraError> {
    if kind == MessageKind::NiProof {
        Err(error(ErrorCode::Usage))
    } else {
        Ok(())
    }
}

/// RFC-0014's `NN-<schema>.tce` name, including the two messages at step 1.
/// NI proofs have no Live filename and return `ERR-USAGE`.
pub fn filename(kind: MessageKind) -> Result<String, TesseraError> {
    let step = match kind {
        MessageKind::SeedCommit => 0,
        MessageKind::Round1 | MessageKind::ProverSeedShare => 1,
        MessageKind::SeedReveal => 2,
        MessageKind::Round2Response => 3,
        MessageKind::ColumnQueries => 4,
        MessageKind::Round3Response => 5,
        MessageKind::NiProof => return Err(error(ErrorCode::Usage)),
    };
    Ok(format!("{step:02}-{kind}.tce"))
}

struct Count(u64);
impl Sink for Count {
    fn put(&mut self, bytes: &[u8]) {
        self.0 = self.0.saturating_add(bytes.len() as u64);
    }
}

fn encoded_len<T: Message>(message: &T, limits: DecodeLimits) -> Result<u64, TesseraError> {
    live(T::KIND)?;
    let mut count = Count((envelope::HEADER_LEN + envelope::DIGEST_LEN) as u64);
    message.encode_to(&mut count);
    check_len(count.0, limits)?;
    Ok(count.0)
}

fn check_len(len: u64, limits: DecodeLimits) -> Result<usize, TesseraError> {
    if len > limits.max_total_bytes {
        return Err(error(ErrorCode::CodecLimit));
    }
    if len < (envelope::HEADER_LEN + envelope::DIGEST_LEN) as u64 {
        return Err(error(ErrorCode::CodecTruncated));
    }
    usize::try_from(len).map_err(|_| error(ErrorCode::CodecLimit))
}

fn read_bytes<R: Read>(mut input: R, len: usize, end: Instant) -> Result<Vec<u8>, TesseraError> {
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 65536];
    while bytes.len() < len {
        remaining(end).map_err(io_error)?;
        let want = chunk.len().min(len - bytes.len());
        input.read_exact(&mut chunk[..want]).map_err(io_error)?;
        remaining(end).map_err(io_error)?;
        if bytes.capacity() - bytes.len() < want {
            // Grow geometrically, only after bytes arrived, and never beyond the
            // checked frame length (INV-TCE-3). A prefix alone allocates no frame.
            let capacity = len.min(bytes.capacity().saturating_mul(2).max(chunk.len()));
            bytes
                .try_reserve_exact(capacity - bytes.len())
                .map_err(|_| error(ErrorCode::ResMemory))?;
        }
        bytes.extend_from_slice(&chunk[..want]);
    }
    remaining(end).map_err(io_error)?;
    Ok(bytes)
}

struct TimedWriter<W> {
    inner: W,
    end: Instant,
}
impl<W: Write> Write for TimedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        remaining(self.end)?;
        let n = self.inner.write(&bytes[..bytes.len().min(65536)])?;
        remaining(self.end)?;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        remaining(self.end)?;
        self.inner.flush()?;
        remaining(self.end)?;
        Ok(())
    }
}

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// A one-session directory exchange with exclusive, atomic message publication.
///
/// The operator owns the directory and transfers complete regular files into it.
/// Final names are never overwritten; symlink and non-file inputs are refused.
/// Filesystem calls have cooperative deadline checks, not kernel-level cancellation.
pub struct DirectoryTransport {
    directory: PathBuf,
    limits: DecodeLimits,
    deadlines: Deadlines,
    failed: bool,
}

impl DirectoryTransport {
    /// Uses an existing directory. Creation, permissions and exchange authentication
    /// are the operator's responsibility, as for an air-gapped transfer.
    pub fn new(
        directory: &Path,
        limits: DecodeLimits,
        deadlines: Deadlines,
    ) -> Result<Self, TesseraError> {
        if !fs::metadata(directory).map_err(io_error)?.is_dir() {
            return Err(error(ErrorCode::Usage));
        }
        Ok(Self {
            directory: directory.to_owned(),
            limits,
            deadlines,
            failed: false,
        })
    }

    /// Publishes precisely the message's envelope, with no stream length prefix.
    /// An existing final name is `ERR-PROTO-ORDER`; it is never replaced.
    pub fn send<T: Message>(&mut self, message: &T) -> Result<(), TesseraError> {
        if self.failed {
            return Err(error(ErrorCode::ProtoOrder));
        }
        let result = self.send_inner(message);
        self.failed = result.is_err();
        result
    }

    fn send_inner<T: Message>(&self, message: &T) -> Result<(), TesseraError> {
        let end = deadline(self.deadlines.for_message(T::KIND))?;
        remaining(end).map_err(io_error)?;
        encoded_len(message, self.limits)?;
        let destination = self.directory.join(filename(T::KIND)?);
        for _ in 0..8 {
            remaining(end).map_err(io_error)?;
            let nonce = getrandom::u64().map_err(|_| error(ErrorCode::ResIo))?;
            let temp = self
                .directory
                .join(format!(".tessera-transport-{nonce:016x}.tmp"));
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let file = match options.open(&temp) {
                Ok(file) => file,
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io_error(e)),
            };
            let _guard = Temporary(temp.clone());
            let mut writer = TimedWriter { inner: file, end };
            envelope::write(&mut writer, &T::KIND.schema(), message).map_err(io_error)?;
            writer.inner.sync_all().map_err(io_error)?;
            remaining(end).map_err(io_error)?;
            fs::hard_link(&temp, destination).map_err(|e| {
                if e.kind() == io::ErrorKind::AlreadyExists {
                    error(ErrorCode::ProtoOrder)
                } else {
                    io_error(e)
                }
            })?;
            File::open(&self.directory)
                .and_then(|dir| dir.sync_all())
                .map_err(io_error)?;
            remaining(end).map_err(io_error)?;
            return Ok(());
        }
        Err(error(ErrorCode::ResIo))
    }

    /// Waits for and decodes one message, bounded before allocation. Every failure
    /// poisons this transport, including a malformed envelope or a timeout.
    pub fn receive<T: Message>(&mut self) -> Result<T, TesseraError> {
        if self.failed {
            return Err(error(ErrorCode::ProtoOrder));
        }
        let result = self.receive_inner::<T>();
        self.failed = result.is_err();
        result
    }

    fn receive_inner<T: Message>(&self) -> Result<T, TesseraError> {
        let end = deadline(self.deadlines.for_message(T::KIND))?;
        let path = self.directory.join(filename(T::KIND)?);
        loop {
            let left = remaining(end).map_err(io_error)?;
            match fs::symlink_metadata(&path) {
                Ok(meta) => {
                    if !meta.is_file() {
                        return Err(error(ErrorCode::ResIo));
                    }
                    let mut options = OpenOptions::new();
                    options.read(true);
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::OpenOptionsExt;
                        // Refuse a symlink/FIFO swapped in after metadata was inspected.
                        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
                    }
                    let file = options.open(&path).map_err(io_error)?;
                    if !file.metadata().map_err(io_error)?.is_file() {
                        return Err(error(ErrorCode::ResIo));
                    }
                    let len = check_len(file.metadata().map_err(io_error)?.len(), self.limits)?;
                    let mut input = file;
                    let bytes = read_bytes(&mut input, len, end)?;
                    let mut extra = [0];
                    if input.read(&mut extra).map_err(io_error)? != 0 {
                        return Err(error(ErrorCode::CodecNoncanonical));
                    }
                    let value = T::open(&bytes, self.limits).map_err(TesseraError::from)?;
                    remaining(end).map_err(io_error)?;
                    return Ok(value);
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    thread::sleep(left.min(Duration::from_millis(5)))
                }
                Err(e) => return Err(io_error(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SeedCommit;

    pub(super) struct Directory(pub PathBuf);
    impl Directory {
        pub(super) fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "tessera-transport-{:016x}",
                getrandom::u64().unwrap()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn transport(&self) -> DirectoryTransport {
            DirectoryTransport::new(
                &self.0,
                DecodeLimits::MESSAGE,
                Deadlines {
                    round2: Duration::from_secs(2),
                    other: Duration::from_secs(2),
                },
            )
            .unwrap()
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    pub(super) fn message() -> SeedCommit {
        SeedCommit {
            session_id: [1; 16],
            commitment: [3; 32],
        }
    }

    #[test]
    fn filenames_and_timeouts_match_the_existing_directory_contract() {
        let names = [
            "00-SeedCommit.tce",
            "01-ProverSeedShare.tce",
            "02-SeedReveal.tce",
            "01-Round1.tce",
            "03-Round2Response.tce",
            "04-ColumnQueries.tce",
            "05-Round3Response.tce",
        ];
        for (kind, name) in MessageKind::ALL[..7].iter().zip(names) {
            assert_eq!(filename(*kind).unwrap(), name);
            assert_eq!(
                Deadlines::default().for_message(*kind),
                Duration::from_secs(if *kind == MessageKind::Round2Response {
                    3600
                } else {
                    600
                })
            );
        }
        assert_eq!(
            filename(MessageKind::NiProof).unwrap_err().code,
            ErrorCode::Usage
        );
    }

    #[test]
    fn directory_publishes_only_canonical_bytes_without_overwriting_or_temporary_files() {
        let dir = Directory::new();
        let mut sender = dir.transport();
        sender.limits.max_total_bytes = message().seal().len() as u64;
        sender.send(&message()).unwrap();
        let path = dir.0.join(filename(MessageKind::SeedCommit).unwrap());
        assert_eq!(fs::read(&path).unwrap(), message().seal());
        let mut receiver = dir.transport();
        receiver.limits.max_total_bytes = message().seal().len() as u64;
        assert_eq!(receiver.receive::<SeedCommit>().unwrap(), message());
        assert_eq!(
            sender.send(&message()).unwrap_err().code,
            ErrorCode::ProtoOrder
        );
        assert_eq!(
            sender.receive::<SeedCommit>().unwrap_err().code,
            ErrorCode::ProtoOrder
        );
        assert_eq!(fs::read(&path).unwrap(), message().seal());
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn directory_refuses_size_before_reading_and_corrupt_digest_and_poisons_failures() {
        let dir = Directory::new();
        let path = dir.0.join(filename(MessageKind::SeedCommit).unwrap());
        let mut bytes = message().seal();
        let mut receiver = dir.transport();
        receiver.limits.max_total_bytes = bytes.len() as u64 - 1;
        fs::write(&path, &bytes).unwrap();
        assert_eq!(
            receiver.receive::<SeedCommit>().unwrap_err().code,
            ErrorCode::CodecLimit
        );
        assert_eq!(
            receiver.receive::<SeedCommit>().unwrap_err().code,
            ErrorCode::ProtoOrder
        );
        assert_eq!(
            receiver.send(&message()).unwrap_err().code,
            ErrorCode::ProtoOrder
        );
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(path, bytes).unwrap();
        assert_eq!(
            dir.transport().receive::<SeedCommit>().unwrap_err().code,
            ErrorCode::CodecDigest
        );
        let mut sender = dir.transport();
        sender.limits.max_total_bytes = 99;
        assert_eq!(
            sender.send(&message()).unwrap_err().code,
            ErrorCode::CodecLimit
        );
    }

    #[test]
    fn an_advertised_length_without_body_bytes_does_not_allocate_the_frame() {
        let info = allocation_counter::measure(|| {
            assert_eq!(
                read_bytes(
                    io::empty(),
                    1 << 20,
                    deadline(Duration::from_secs(1)).unwrap()
                )
                .unwrap_err()
                .code,
                ErrorCode::ResIo
            );
        });
        assert!(
            info.bytes_max <= 65536,
            "allocated ahead of received bytes: {info:?}"
        );
    }

    #[test]
    fn directory_waits_for_atomic_publication_and_times_out_when_absent() {
        let dir = Directory::new();
        let mut receiver = dir.transport();
        receiver.deadlines.other = Duration::from_millis(20);
        assert_eq!(
            receiver.receive::<SeedCommit>().unwrap_err().code,
            ErrorCode::ProtoTimeout
        );
        let mut sender = dir.transport();
        let task = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            sender.send(&message()).unwrap();
        });
        assert_eq!(dir.transport().receive::<SeedCommit>().unwrap(), message());
        task.join().unwrap();
    }

    #[test]
    fn directory_times_out_even_for_a_ready_file_and_refuses_nonfiles() {
        let dir = Directory::new();
        let path = dir.0.join(filename(MessageKind::SeedCommit).unwrap());
        let mut receiver = dir.transport();
        receiver.deadlines.other = Duration::ZERO;
        fs::write(&path, message().seal()).unwrap();
        assert_eq!(
            receiver.receive::<SeedCommit>().unwrap_err().code,
            ErrorCode::ProtoTimeout
        );
        let mut sender = dir.transport();
        sender.deadlines.other = Duration::ZERO;
        assert_eq!(
            sender.send(&message()).unwrap_err().code,
            ErrorCode::ProtoTimeout
        );
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert_eq!(
            dir.transport().receive::<SeedCommit>().unwrap_err().code,
            ErrorCode::ResIo
        );
        fs::remove_dir(&path).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("not-present", &path).unwrap();
            assert_eq!(
                dir.transport().receive::<SeedCommit>().unwrap_err().code,
                ErrorCode::ResIo
            );
        }
    }
}
