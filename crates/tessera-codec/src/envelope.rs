// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The artifact envelope (RFC-0003 'Envelope'): a fixed header, the TCE payload, and a BLAKE3
//! digest of both. The digest trails the payload, so a writer can stream, and a reader checks
//! the digest before it decodes anything.
//!
//! ```text
//! offset  field             value
//!      0  magic             b"TSRA"
//!      4  envelope_version  1 (u16)
//!      6  schema_id         03 §5 (u16)
//!      8  schema_version    (u16)
//!     10  reserved          0 (u16)
//!     12  payload_len       (u64)
//!     20  payload           payload_len bytes
//!  20+len digest            BLAKE3 derive_key("tessera/v1/envelope") over all preceding bytes
//! ```
//!
//! Every reader checks an envelope in the order RFC-0003 'Envelope' fixes, so that independent
//! readers return the same code at the same offset for every input. Offsets count from the
//! start of the envelope.
//! 1. The header fields, each as it is read: the magic (`ERR-CODEC-MAGIC`), the envelope version
//!    (`ERR-CODEC-VERSION`), the schema id the reader expects (`ERR-CODEC-MAGIC`), a schema
//!    version it accepts (`ERR-CODEC-VERSION`), a zero `reserved` (`ERR-CODEC-NONCANONICAL`),
//!    and a `payload_len` that keeps the whole envelope within `max_total_bytes`
//!    (`ERR-CODEC-LIMIT`). A field cut short is `ERR-CODEC-TRUNCATED` at its offset.
//! 2. The length of the rest: fewer than `payload_len + 32` bytes is `ERR-CODEC-TRUNCATED` at
//!    offset 12, more is `ERR-CODEC-NONCANONICAL` at the first byte past the digest.
//! 3. The digest (`ERR-CODEC-DIGEST`, at its offset).
//! 4. The payload, decoded as the schema version found, which it must fill (INV-TCE-4).
//!
//! [`write_file`] writes an artifact by atomic rename, so that its path holds the previous
//! file or the whole new one, never part of one, and gives a secret artifact mode 0600 from the
//! moment it exists (06 §7).

use core::fmt;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use zeroize::Zeroizing;

use crate::error::{ErrorCode, SchemaId, TesseraError};
use crate::schema::{Class, Schema};
use crate::{Decode, DecodeError, DecodeLimits, Encode, Sink, decode};

/// The first four bytes of every envelope.
pub const MAGIC: [u8; 4] = *b"TSRA";

/// The envelope layout this crate reads and writes.
pub const ENVELOPE_VERSION: u16 = 1;

/// The header's length.
pub const HEADER_LEN: usize = 20;

/// The digest's length.
pub const DIGEST_LEN: usize = 32;

/// The BLAKE3 key-derivation context of the digest: the `envelope` domain of RFC-0004.
pub const DIGEST_CONTEXT: &str = "tessera/v1/envelope";

/// Where the schema version sits.
const AT_SCHEMA_VERSION: u64 = 8;

/// Where the payload length sits, which is where a short envelope is reported.
const AT_PAYLOAD_LEN: u64 = 12;

/// The unit in which payloads are hashed and copied while streaming. The buffers that hold a
/// chunk, and the hasher whose block buffer holds up to 63 of its bytes, are zeroized when
/// dropped: a payload may be secret-class data (05 §3, OBS-2, OBS-6).
const CHUNK: usize = 64 << 10;

/// An envelope digest.
pub type Digest = [u8; DIGEST_LEN];

/// What an envelope's header declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Header {
    /// The payload's schema.
    pub schema: SchemaId,
    /// The schema version the payload follows.
    pub version: u16,
    /// The payload's length in bytes.
    pub payload_len: u64,
}

impl Header {
    /// The header's bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut bytes = [0u8; HEADER_LEN];
        bytes[..4].copy_from_slice(&MAGIC);
        bytes[4..6].copy_from_slice(&ENVELOPE_VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.schema.0.to_le_bytes());
        bytes[8..10].copy_from_slice(&self.version.to_le_bytes());
        // Bytes 10 and 11 hold `reserved`, which is zero.
        bytes[12..].copy_from_slice(&self.payload_len.to_le_bytes());
        bytes
    }

    /// The offset of the digest.
    fn digest_at(&self) -> u64 {
        HEADER_LEN as u64 + self.payload_len
    }
}

/// The envelope of `value`, as the current version of `schema`.
pub fn seal<T: Encode + ?Sized>(schema: &Schema, value: &T) -> Vec<u8> {
    // Sized first, so the buffer never grows: growing would leave copies of a secret-class
    // payload behind in freed memory.
    let mut size = Count(0);
    value.encode_to(&mut size);
    let capacity = usize::try_from(size.0).map_or(0, |n| n.saturating_add(HEADER_LEN + DIGEST_LEN));
    let mut out = Vec::with_capacity(capacity);
    out.resize(HEADER_LEN, 0);
    value.encode_to(&mut out);
    let header = Header {
        schema: schema.id,
        version: schema.version,
        payload_len: (out.len() - HEADER_LEN) as u64,
    };
    out[..HEADER_LEN].copy_from_slice(&header.to_bytes());
    let digest = hasher().update(&out).finalize();
    out.extend_from_slice(digest.as_bytes());
    out
}

/// Writes the envelope of `value`, as the current version of `schema`, while encoding it: a
/// first pass sizes the payload for the header, and a second writes the payload and hashes it on
/// the way, so memory stays constant whatever the payload's size. Returns the digest.
///
/// Byte sequences encode with one call each, so the sizing pass costs little next to the
/// writing pass. After an error, `out` holds part of an envelope, which the caller discards.
pub fn write<T: Encode + ?Sized, W: Write>(
    out: &mut W,
    schema: &Schema,
    value: &T,
) -> io::Result<Digest> {
    let mut size = Count(0);
    value.encode_to(&mut size);
    let header = Header {
        schema: schema.id,
        version: schema.version,
        payload_len: size.0,
    };
    let mut tee = Tee {
        out,
        hasher: hasher(),
        buffer: Zeroizing::new(Vec::with_capacity(chunk_for(header.digest_at()))),
        written: 0,
        error: None,
    };
    tee.put(&header.to_bytes());
    value.encode_to(&mut tee);
    let (written, digest) = tee.finish()?;
    if written != header.digest_at() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the payload encoded to different lengths in the sizing and writing passes",
        ));
    }
    out.write_all(&digest)?;
    Ok(digest)
}

/// Writes the envelope of `value`, as the current version of `schema`, to the file at `path` by
/// atomic rename (06 §7), and returns its digest:
///
/// 1. a new file in `path`'s directory, `.<name>.<16 hex digits>.tmp`, its digits drawn from the
///    operating system's CSPRNG, so that no other process can know the name beforehand. It is
///    created exclusively: a name another file or a symbolic link already holds is never opened
///    or followed, but drawn again, up to [`TEMPORARY_ATTEMPTS`] times. Its mode, set as it is
///    created and before any byte is written, is 0600 when `schema` is secret-class and the
///    default mode otherwise;
/// 2. the envelope written to it as [`write()`] writes one, through a buffer that is cleared when
///    it is freed, then flushed to disk;
/// 3. the file renamed over `path`, then the directory flushed, so that the rename is durable.
///
/// The file is never readable by others, even while it is written, and `path` holds the
/// previous file or the whole new one. If any step fails, or encoding the value panics, the new
/// file is removed and `path` is left as it was; an error after the rename, in flushing the
/// directory, is returned with the new file in place.
///
/// A process that stops between steps 1 and 3, by a crash or a kill, leaves its new file behind,
/// beside `path`: `.<name>.<16 hex digits>.tmp`, of mode 0600 for a secret-class artifact, holding
/// some or all of the envelope. No reader opens it, and it may be deleted.
///
/// # Errors
///
/// The operating system's: a directory that does not exist or cannot be written, a full disk,
/// a CSPRNG that cannot be read. When every name drawn is taken, [`io::ErrorKind::AlreadyExists`]
/// with a [`TemporaryNamesTaken`] inside. Only Unix systems set a mode, so a secret-class
/// artifact elsewhere is [`io::ErrorKind::Unsupported`].
pub fn write_file<T: Encode + ?Sized>(
    path: &Path,
    schema: &Schema,
    value: &T,
) -> io::Result<Digest> {
    replace(path, schema.class == Class::Secret, |file| {
        write(file, schema, value)
    })
}

/// How many names [`write_file`] draws for its new file before it gives up. With 64 random bits
/// a name, a draw meets an existing file only when the directory holds billions of them, or when
/// the CSPRNG repeats.
pub const TEMPORARY_ATTEMPTS: u32 = 8;

/// Every name [`write_file`] drew for its new file was taken, by a file or a symbolic link: the
/// error inside the [`io::ErrorKind::AlreadyExists`] it then returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemporaryNamesTaken {
    /// The names drawn, each taken: [`TEMPORARY_ATTEMPTS`].
    pub attempts: u32,
}

impl fmt::Display for TemporaryNamesTaken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the {} names drawn for a new file beside the target were all taken",
            self.attempts
        )
    }
}

impl Error for TemporaryNamesTaken {}

/// The procedure of [`write_file`], with `contents` writing the new file.
fn replace<R>(
    path: &Path,
    secret: bool,
    contents: impl FnOnce(&mut File) -> io::Result<R>,
) -> io::Result<R> {
    let mut draw = || getrandom::u64().map_err(io::Error::from);
    replace_with(path, secret, &mut draw, contents)
}

/// [`replace`], with the random part of each name the new file may take from `draw`.
fn replace_with<R>(
    path: &Path,
    secret: bool,
    draw: &mut dyn FnMut() -> io::Result<u64>,
    contents: impl FnOnce(&mut File) -> io::Result<R>,
) -> io::Result<R> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the path names no file"))?;
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let mut options = OpenOptions::new();
    // Exclusive creation: an existing file or link at the name fails the open, and nothing is
    // followed.
    options.write(true).create_new(true);
    if secret {
        restrict(&mut options)?;
    }
    let (mut file, temporary) = 'open: {
        for _ in 0..TEMPORARY_ATTEMPTS {
            let mut temporary = OsString::from(".");
            temporary.push(name);
            temporary.push(format!(".{:016x}.tmp", draw()?));
            let temporary = directory.join(temporary);
            match options.open(&temporary) {
                Ok(file) => break 'open (file, temporary),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        }
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            TemporaryNamesTaken {
                attempts: TEMPORARY_ATTEMPTS,
            },
        ));
    };
    let guard = Removal(Some(temporary));
    let written = contents(&mut file)?;
    file.sync_all()?;
    drop(file);
    let temporary = guard.disarm();
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    File::open(directory)?.sync_all()?;
    Ok(written)
}

/// Mode 0600 for a file about to be created: readable and writable by its owner alone.
#[cfg(unix)]
fn restrict(options: &mut OpenOptions) -> io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    options.mode(0o600);
    Ok(())
}

/// No mode exists to set, so a secret-class artifact is not written.
#[cfg(not(unix))]
fn restrict(_: &mut OpenOptions) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "secret-class artifacts are written with mode 0600, which only Unix systems have",
    ))
}

/// Removes a new file when it drops, unless disarmed: after an error, and during a panic's
/// unwinding, the file goes.
struct Removal(Option<PathBuf>);

impl Removal {
    /// The path, no longer removed.
    fn disarm(mut self) -> PathBuf {
        self.0.take().unwrap_or_default()
    }
}

impl Drop for Removal {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_file(path);
        }
    }
}

/// Checks an envelope of `schema` held in memory, in the order of the module documentation,
/// and returns its header and payload once the digest verifies. The reader accepts the current
/// version of `schema` and the one before; decode the payload by [`Header::version`].
pub fn open_payload<'a>(
    bytes: &'a [u8],
    schema: &Schema,
    limits: DecodeLimits,
) -> Result<(Header, &'a [u8]), DecodeError> {
    let header = check_header(&bytes[..bytes.len().min(HEADER_LEN)], schema, limits)?;
    let rest = bytes.get(HEADER_LEN..).unwrap_or_default();
    // No overflow: the header check bounded the whole envelope by `max_total_bytes`.
    let expected = header.payload_len + DIGEST_LEN as u64;
    let found = rest.len() as u64;
    if found < expected {
        return Err(error(ErrorCode::CodecTruncated, AT_PAYLOAD_LEN));
    }
    if found > expected {
        return Err(error(
            ErrorCode::CodecNoncanonical,
            HEADER_LEN as u64 + expected,
        ));
    }
    let (payload, digest) = rest.split_at(rest.len() - DIGEST_LEN);
    let computed = hasher()
        .update(&bytes[..HEADER_LEN])
        .update(payload)
        .finalize();
    if computed.as_bytes() != digest {
        return Err(error(ErrorCode::CodecDigest, header.digest_at()));
    }
    Ok((header, payload))
}

/// Opens an envelope of the current version of `schema` and decodes its payload as `T`, which
/// must fill it (INV-TCE-4). Errors in the payload carry offsets from the start of the envelope.
///
/// `T` is one version's layout, so the previous version, which [`open_payload`] accepts, is
/// `ERR-CODEC-VERSION` here.
pub fn open<T: Decode>(
    bytes: &[u8],
    schema: &Schema,
    limits: DecodeLimits,
) -> Result<T, DecodeError> {
    let (header, payload) = open_payload(bytes, schema, limits)?;
    if header.version != schema.version {
        return Err(error(ErrorCode::CodecVersion, AT_SCHEMA_VERSION));
    }
    decode(payload, limits).map_err(|e| error(e.code, HEADER_LEN as u64 + e.offset))
}

/// Checks an envelope of `schema` read from `input`, a file for instance, without decoding it
/// and in constant memory: the checks of [`open_payload`], in the same order and with the same
/// codes and offsets. It holds at most 64 KiB of the input at a time.
///
/// Verification covers the bytes read here. Decode only bytes that were verified: read the
/// envelope again and open it with [`open_payload`], which checks the digest again.
pub fn verify<R: Read>(
    mut input: R,
    schema: &Schema,
    limits: DecodeLimits,
) -> Result<Header, ReadError> {
    let mut head = [0u8; HEADER_LEN];
    let got = fill(&mut input, &mut head)?;
    let header = check_header(&head[..got], schema, limits)?;
    let mut hasher = hasher();
    hasher.update(&head);
    let short = || ReadError::Invalid(error(ErrorCode::CodecTruncated, AT_PAYLOAD_LEN));
    let mut buffer = Zeroizing::new(vec![0u8; chunk_for(header.payload_len)]);
    let mut left = header.payload_len;
    while left > 0 {
        let want = usize::try_from(left).map_or(buffer.len(), |n| n.min(buffer.len()));
        let got = fill(&mut input, &mut buffer[..want])?;
        hasher.update(&buffer[..got]);
        if got < want {
            return Err(short());
        }
        left -= got as u64;
    }
    let mut digest = [0u8; DIGEST_LEN];
    if fill(&mut input, &mut digest)? < DIGEST_LEN {
        return Err(short());
    }
    if fill(&mut input, &mut [0u8; 1])? > 0 {
        let past = header.digest_at() + DIGEST_LEN as u64;
        return Err(error(ErrorCode::CodecNoncanonical, past).into());
    }
    if hasher.finalize().as_bytes() != &digest {
        return Err(error(ErrorCode::CodecDigest, header.digest_at()).into());
    }
    Ok(header)
}

/// Why [`verify`] failed.
#[derive(Debug)]
pub enum ReadError {
    /// The envelope breaks RFC-0003.
    Invalid(DecodeError),
    /// Reading the input failed.
    Io(io::Error),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(e) => e.fmt(f),
            Self::Io(e) => write!(f, "reading the envelope failed: {e}"),
        }
    }
}

impl Error for ReadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Invalid(e) => Some(e),
            Self::Io(e) => Some(e),
        }
    }
}

impl From<DecodeError> for ReadError {
    fn from(error: DecodeError) -> Self {
        Self::Invalid(error)
    }
}

impl From<io::Error> for ReadError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// `ERR-RES-IO` for a failed read, the envelope's code otherwise.
impl From<ReadError> for TesseraError {
    fn from(error: ReadError) -> Self {
        match error {
            ReadError::Invalid(e) => e.into(),
            ReadError::Io(_) => TesseraError::new(ErrorCode::ResIo),
        }
    }
}

fn hasher() -> Zeroizing<blake3::Hasher> {
    Zeroizing::new(blake3::Hasher::new_derive_key(DIGEST_CONTEXT))
}

/// A buffer size for `len` bytes: one chunk at most.
fn chunk_for(len: u64) -> usize {
    usize::try_from(len).map_or(CHUNK, |len| len.min(CHUNK))
}

fn error(code: ErrorCode, offset: u64) -> DecodeError {
    DecodeError { code, offset }
}

/// The `N` bytes at `at`, or `ERR-CODEC-TRUNCATED` there.
fn field<const N: usize>(prefix: &[u8], at: usize) -> Result<[u8; N], DecodeError> {
    prefix
        .get(at..at + N)
        .and_then(|bytes| <[u8; N]>::try_from(bytes).ok())
        .ok_or(error(ErrorCode::CodecTruncated, at as u64))
}

/// Checks the header fields at the start of `prefix`, each as it is read (step 1 of the module
/// documentation).
fn check_header(
    prefix: &[u8],
    schema: &Schema,
    limits: DecodeLimits,
) -> Result<Header, DecodeError> {
    let u16_at = |at| field(prefix, at).map(u16::from_le_bytes);
    if field::<4>(prefix, 0)? != MAGIC {
        return Err(error(ErrorCode::CodecMagic, 0));
    }
    if u16_at(4)? != ENVELOPE_VERSION {
        return Err(error(ErrorCode::CodecVersion, 4));
    }
    let id = SchemaId(u16_at(6)?);
    if id != schema.id {
        return Err(error(ErrorCode::CodecMagic, 6));
    }
    let version = u16_at(8)?;
    if !schema.reads(version) {
        return Err(error(ErrorCode::CodecVersion, AT_SCHEMA_VERSION));
    }
    if u16_at(10)? != 0 {
        return Err(error(ErrorCode::CodecNoncanonical, 10));
    }
    let payload_len = u64::from_le_bytes(field(prefix, 12)?);
    let whole = payload_len.checked_add((HEADER_LEN + DIGEST_LEN) as u64);
    if whole.is_none_or(|whole| whole > limits.max_total_bytes) {
        return Err(error(ErrorCode::CodecLimit, AT_PAYLOAD_LEN));
    }
    Ok(Header {
        schema: id,
        version,
        payload_len,
    })
}

/// Reads until `buffer` is full or the input ends, and returns how many bytes it read. The first
/// end of the input is final: nothing after it is read.
fn fill<R: Read>(input: &mut R, buffer: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        let room = buffer.len() - filled;
        match input.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(n) if n <= room => filled += n,
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "the reader reported more bytes than it was given room for",
                ));
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// Counts the bytes of an encoding.
struct Count(u64);

impl Sink for Count {
    fn put(&mut self, bytes: &[u8]) {
        self.0 += bytes.len() as u64;
    }
}

/// Writes an encoding and hashes it, through a buffer, keeping the first write error.
struct Tee<'w, W: Write> {
    out: &'w mut W,
    hasher: Zeroizing<blake3::Hasher>,
    buffer: Zeroizing<Vec<u8>>,
    written: u64,
    error: Option<io::Error>,
}

impl<W: Write> Tee<'_, W> {
    fn emit(&mut self, bytes: &[u8]) {
        emit(&mut *self.out, &mut self.hasher, &mut self.error, bytes);
    }

    fn flush_buffer(&mut self) {
        emit(
            &mut *self.out,
            &mut self.hasher,
            &mut self.error,
            &self.buffer,
        );
        self.buffer.clear();
    }

    /// The bytes written and their digest, or the first write error.
    fn finish(mut self) -> io::Result<(u64, Digest)> {
        self.flush_buffer();
        match self.error {
            Some(e) => Err(e),
            None => Ok((self.written, *self.hasher.finalize().as_bytes())),
        }
    }
}

/// Hashes `bytes` and writes them to `out`, unless an earlier write failed.
fn emit<W: Write>(
    out: &mut W,
    hasher: &mut blake3::Hasher,
    error: &mut Option<io::Error>,
    bytes: &[u8],
) {
    if error.is_none() {
        hasher.update(bytes);
        if let Err(e) = out.write_all(bytes) {
            *error = Some(e);
        }
    }
}

impl<W: Write> Sink for Tee<'_, W> {
    fn put(&mut self, bytes: &[u8]) {
        self.written += bytes.len() as u64;
        if self.buffer.len() + bytes.len() > CHUNK {
            self.flush_buffer();
        }
        if bytes.len() >= CHUNK {
            self.emit(bytes);
        } else {
            self.buffer.extend_from_slice(bytes);
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::io::Write as _;
    use std::os::unix::fs::{PermissionsExt, symlink};

    use super::*;

    /// A fresh scratch directory under the system's temporary directory.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tessera-codec-envelope-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("a scratch directory");
        dir
    }

    /// A symbolic link planted at the first name drawn is neither opened nor followed: the name
    /// is drawn again, and the write goes to a new file of mode 0600.
    #[test]
    fn a_link_planted_at_a_drawn_name_is_not_followed_and_another_name_is_drawn() {
        let dir = scratch("planted");
        let target = dir.join("state.twc");
        let elsewhere = dir.join("elsewhere");
        let planted = dir.join(format!(".state.twc.{:016x}.tmp", 1));
        symlink(&elsewhere, &planted).expect("a link");
        let mut draws = [1u64, 2].into_iter();
        let mut draw = || -> io::Result<u64> { Ok(draws.next().expect("two draws")) };
        replace_with(&target, true, &mut draw, |file| file.write_all(b"new")).expect("written");
        assert_eq!(fs::read(&target).expect("the target"), b"new");
        let mode = fs::metadata(&target)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
        assert!(!elsewhere.exists(), "nothing was written through the link");
        let link = fs::symlink_metadata(&planted).expect("the link");
        assert!(link.file_type().is_symlink(), "the link is left as it was");
        let _ = fs::remove_dir_all(&dir);
    }

    /// When every name drawn is taken, the write fails with `AlreadyExists` and a
    /// `TemporaryNamesTaken`, after exactly `TEMPORARY_ATTEMPTS` draws, and changes nothing.
    #[test]
    fn when_every_name_drawn_is_taken_the_write_fails_with_a_typed_error() {
        let dir = scratch("taken");
        let target = dir.join("state.twc");
        fs::write(&target, b"previous").expect("a previous file");
        let elsewhere = dir.join("elsewhere");
        symlink(&elsewhere, dir.join(format!(".state.twc.{:016x}.tmp", 7))).expect("a link");
        let mut draws = 0;
        let mut draw = || -> io::Result<u64> {
            draws += 1;
            Ok(7)
        };
        let error = replace_with(&target, true, &mut draw, |file| file.write_all(b"new"))
            .expect_err("every name is taken");
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        let inner = error
            .get_ref()
            .and_then(|e| e.downcast_ref::<TemporaryNamesTaken>());
        assert_eq!(
            inner,
            Some(&TemporaryNamesTaken {
                attempts: TEMPORARY_ATTEMPTS
            })
        );
        assert_eq!(draws, TEMPORARY_ATTEMPTS);
        assert_eq!(fs::read(&target).expect("the target"), b"previous");
        assert!(!elsewhere.exists(), "nothing was written through the link");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A CSPRNG that cannot be read fails the write before any file is created.
    #[test]
    fn a_draw_that_fails_fails_the_write_before_any_file_exists() {
        let dir = scratch("draw");
        let target = dir.join("state.twc");
        let mut draw = || -> io::Result<u64> { Err(io::Error::other("no entropy")) };
        let error = replace_with(&target, true, &mut draw, |file| file.write_all(b"new"))
            .expect_err("no name");
        assert_eq!(error.to_string(), "no entropy");
        assert_eq!(fs::read_dir(&dir).expect("the directory").count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }
}
