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

use core::fmt;
use std::error::Error;
use std::io::{self, Read, Write};

use zeroize::Zeroizing;

use crate::error::{ErrorCode, SchemaId, TesseraError};
use crate::schema::Schema;
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
/// chunk are zeroized when dropped: a payload may be secret-class data (05 §3, OBS-2).
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

fn hasher() -> blake3::Hasher {
    blake3::Hasher::new_derive_key(DIGEST_CONTEXT)
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
    hasher: blake3::Hasher,
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
