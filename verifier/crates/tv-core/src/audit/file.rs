// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The ledger's file (RFC-0067 'Updates', 'One writer'): one regular file of one name, held by
//! one writer under a lock file, read without following a link, and written atomically, its
//! directory synced as part of every write.
//!
//! - **The path.** The ledger is a regular file, neither a symbolic link nor a file of several
//!   hard links, else `ERR-USAGE`: two names for one ledger would give two locks, and two writers.
//!   Its metadata is read without following a link, so a FIFO or a device is refused before it
//!   is opened; the file is then opened without following a link and without blocking, where the
//!   host has the flags for it, and compared, by device and inode, with the one checked. Its
//!   directory is resolved once, when the ledger is opened, and every later name is built in it.
//! - **One writer.** `<name>.lock`, created exclusively in that directory, holds a token naming
//!   its writer and keeps a second writer out while the first holds the ledger. A lock left by a
//!   crash refuses later writers with `ERR-RES-IO` until the operator, having checked that no
//!   verifier runs, removes it. A writer removes only a lock that still holds its token.
//! - **Two windows remain**, each a few instructions wide, until the kernel holds the lock for
//!   the writer (#534, `std` from Rust 1.89, above the MSRV): between the token's check and the
//!   lock's removal ([`Held::remove`]), and between a write's check and its rename
//!   ([`write_atomic`]). A writer that stalls in either while an operator clears its lock and
//!   another writer runs can remove the other's lock, or rename over the other's record.
//! - **A write.** The new ledger goes to a fresh temporary file beside the old one, which is
//!   synced and renamed over it, and then the directory is synced. A failed directory sync fails
//!   the write with `ERR-RES-IO`: after a power loss the old ledger could come back, and with it
//!   a request accepted again (INV-AUD-3). Before the rename, the lock must still hold this
//!   writer's token and the ledger must still be the file this writer last read or wrote, by
//!   device and inode, else `ERR-RES-IO` and nothing is written: a writer that stalled while an
//!   operator cleared its lock could otherwise rename a stale ledger over another writer's
//!   record. A device and inode can be reused once a file is removed, so the lock's token, unique
//!   to its writer, carries the lock's half of the check. The ledger must also still be one regular file of one name. The ledger in memory
//!   changes only once the write has succeeded, so a failed write leaves the caller's verdict
//!   short of `Accept`.
//! - **One limit.** Every reader and writer of the ledger uses [`LEDGER_LIMITS`], `tv-codec`'s
//!   message limits, so a ledger a writer accepts is one every reader opens (RFC-0067 'One
//!   limit').

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tv_codec::DecodeLimits;

use super::ledger::VerifierLedger;
use super::sample::AuditSample;
use crate::anchor::Failure;
use crate::protocol::{FixedBits, PublicStatement};
use crate::verdict::{ErrorCode as Code, VerifierError};

/// The decode limits of the ledger, for every reader and writer: `tv-codec`'s message limits
/// (RFC-0067 'One limit'). No write makes the ledger larger than they admit.
pub const LEDGER_LIMITS: DecodeLimits = DecodeLimits::MESSAGE;

const fn io() -> VerifierError {
    VerifierError { code: Code::Io }
}

const fn misuse() -> VerifierError {
    VerifierError { code: Code::Usage }
}

// ----- The path --------------------------------------------------------------------------------

/// What names a file on the host: its device and inode on Unix, nothing elsewhere.
type FileId = (u64, u64);

#[cfg(unix)]
fn file_id(meta: &fs::Metadata) -> FileId {
    use std::os::unix::fs::MetadataExt;
    (meta.dev(), meta.ino())
}

#[cfg(not(unix))]
fn file_id(_: &fs::Metadata) -> FileId {
    (0, 0)
}

/// Whether the file has one name. Hosts without link counts in `std` count one.
#[cfg(unix)]
fn one_name(meta: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    meta.nlink() == 1
}

#[cfg(not(unix))]
fn one_name(_: &fs::Metadata) -> bool {
    true
}

/// `O_NOFOLLOW | O_NONBLOCK` on the targets whose values can be cited, so that a path swapped for
/// a link or a FIFO after its check fails to open, or opens without blocking, rather than being
/// followed or waited on:
///
/// - Linux's generic layout, `include/uapi/asm-generic/fcntl.h`, which x86, x86_64 and riscv64
///   use: `O_NOFOLLOW` `0o400000`, `O_NONBLOCK` `0o4000`;
/// - Linux on arm and aarch64, `arch/arm/include/uapi/asm/fcntl.h` and
///   `arch/arm64/include/uapi/asm/fcntl.h`: `O_NOFOLLOW` `0o100000`, where `0o400000` is
///   `O_LARGEFILE`, and `O_NONBLOCK` `0o4000`;
/// - Apple's systems, `sys/fcntl.h`: `O_NOFOLLOW` `0x0100`, `O_NONBLOCK` `0x0004`.
///
/// Every other target opens with no flag, as a host without them would: the check by metadata
/// that does not follow a link, before the open, and the comparison of the opened file's device
/// and inode with it after, are then the only guards against a swap between the two.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "x86", target_arch = "riscv64")
))]
const NO_FOLLOW_NO_BLOCK: i32 = 0o400_000 | 0o4_000;
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "arm")))]
const NO_FOLLOW_NO_BLOCK: i32 = 0o100_000 | 0o4_000;
#[cfg(target_vendor = "apple")]
const NO_FOLLOW_NO_BLOCK: i32 = 0x0100 | 0x0004;
#[cfg(not(any(
    all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "x86",
            target_arch = "riscv64",
            target_arch = "aarch64",
            target_arch = "arm"
        )
    ),
    target_vendor = "apple"
)))]
const NO_FOLLOW_NO_BLOCK: i32 = 0;

/// Whether the host's flags are named, so that an open refuses a link.
#[cfg(test)]
const HAS_NO_FOLLOW: bool = NO_FOLLOW_NO_BLOCK != 0;

/// Opens `path` for reading without following a link and without blocking, as far as the host
/// allows ([`NO_FOLLOW_NO_BLOCK`]).
fn open_quietly(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(NO_FOLLOW_NO_BLOCK);
    }
    options.open(path)
}

/// RFC-0067 'The path': `path` names a regular file of one name, read from metadata that does
/// not follow a link, so that nothing is opened yet.
///
/// # Errors
///
/// `ERR-RES-IO` when nothing can be read at `path`; `ERR-USAGE` for a symbolic link, anything
/// but a regular file, or a file of several hard links.
fn checked(path: &Path) -> Result<fs::Metadata, VerifierError> {
    let meta = fs::symlink_metadata(path).map_err(|_| io())?;
    if !meta.file_type().is_file() || !one_name(&meta) {
        return Err(misuse());
    }
    Ok(meta)
}

/// Reads the regular file `path`, at most `limit` bytes, after [`checked`], and checks that the
/// file opened is the one checked; returns the bytes and the file's identity.
fn read_regular(path: &Path, limit: u64) -> Result<(Vec<u8>, FileId), VerifierError> {
    read_regular_with(path, limit, &|| ())
}

/// [`read_regular`], with `between` run after the check and before the open, where a test makes
/// the file grow.
fn read_regular_with(
    path: &Path,
    limit: u64,
    between: &dyn Fn(),
) -> Result<(Vec<u8>, FileId), VerifierError> {
    let before = checked(path)?;
    if before.len() > limit {
        return Err(VerifierError {
            code: Code::CodecLimit,
        });
    }
    between();
    let mut file = open_quietly(path).map_err(|_| io())?;
    let after = file.metadata().map_err(|_| io())?;
    if !after.is_file() || !one_name(&after) || file_id(&after) != file_id(&before) {
        // Replaced between the check and the open.
        return Err(misuse());
    }
    // The file may have grown since it was checked: its size is checked again before anything
    // is reserved for it.
    if after.len() > limit {
        return Err(VerifierError {
            code: Code::CodecLimit,
        });
    }
    let mut out = Vec::new();
    out.try_reserve_exact(usize::try_from(after.len()).map_err(|_| io())?)
        .map_err(|_| VerifierError { code: Code::Memory })?;
    // One byte past the limit shows a file that grew after its size was read.
    Read::by_ref(&mut file)
        .take(limit.saturating_add(1))
        .read_to_end(&mut out)
        .map_err(|_| io())?;
    if u64::try_from(out.len()).map_or(true, |n| n > limit) {
        return Err(VerifierError {
            code: Code::CodecLimit,
        });
    }
    Ok((out, file_id(&after)))
}

/// Reads the ledger at `path`, a snapshot for reading only: the file as its last atomic write
/// left it, decoded under [`LEDGER_LIMITS`], the limits every writer keeps it within. It takes no
/// lock, since a write replaces the file whole; it applies the path's rules, as every reader of
/// the ledger does.
///
/// # Errors
///
/// `ERR-RES-IO` for a file that cannot be read; `ERR-USAGE` for a symbolic link, a file that is
/// not regular, or one of several hard links; `ERR-CODEC-LIMIT` for a file larger than the
/// limits; and the decoder's errors.
pub fn read_ledger(path: &Path) -> Result<VerifierLedger, VerifierError> {
    read_ledger_under(path, &LEDGER_LIMITS).map(|(ledger, _)| ledger)
}

/// [`read_ledger`] under `limits`, with the identity of the file read.
fn read_ledger_under(
    path: &Path,
    limits: &DecodeLimits,
) -> Result<(VerifierLedger, FileId), VerifierError> {
    let (bytes, id) = read_regular(path, limits.max_total_bytes)?;
    let ledger = VerifierLedger::open(&bytes, limits).map_err(|e| VerifierError {
        code: e.code().into(),
    })?;
    Ok((ledger, id))
}

/// A ledger's place: its directory, resolved once, and its name in it.
#[derive(Clone, Debug)]
struct Place {
    dir: PathBuf,
    name: OsString,
}

impl Place {
    fn resolve(path: &Path) -> Result<Self, VerifierError> {
        let name = path.file_name().ok_or(misuse())?.to_os_string();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let dir = fs::canonicalize(parent).map_err(|_| io())?;
        Ok(Self { dir, name })
    }

    fn file(&self) -> PathBuf {
        self.dir.join(&self.name)
    }

    fn beside(&self, prefix: &str, suffix: &str) -> PathBuf {
        let mut name = OsString::from(prefix);
        name.push(&self.name);
        name.push(suffix);
        self.dir.join(name)
    }
}

// ----- The writer ------------------------------------------------------------------------------

/// The verifier's ledger file, held for writing (RFC-0067 'The file', 'One writer'): a lock file
/// beside it, `<name>.lock`, created exclusively and holding this writer's token, keeps a second
/// writer out until this value is dropped, so that two sessions cannot both accept one request
/// (INV-AUD-3). Every update is written atomically, the directory synced, and only while the lock
/// still holds the token and the ledger is still the file this writer last read or wrote; the
/// ledger in memory changes only once the write succeeds.
#[derive(Debug)]
pub struct LedgerFile {
    place: Place,
    lock: Held,
    ledger: VerifierLedger,
    limits: DecodeLimits,
    /// The ledger file this writer last read or wrote.
    file: FileId,
}

impl LedgerFile {
    /// Creates a ledger file at `path`, holding an empty ledger, and holds it.
    ///
    /// # Errors
    ///
    /// `ERR-USAGE` when anything exists at `path`, whose ledger would be lost; `ERR-RES-IO` when
    /// the lock is held or lost, the directory cannot be resolved or a file cannot be written or
    /// synced; `ERR-RES-MEMORY`.
    pub fn create(path: &Path) -> Result<Self, VerifierError> {
        Self::create_with(path, &LEDGER_LIMITS, &REAL)
    }

    fn create_with(
        path: &Path,
        limits: &DecodeLimits,
        hooks: &Hooks<'_>,
    ) -> Result<Self, VerifierError> {
        let place = Place::resolve(path)?;
        let lock = Lock::take(&place)?;
        // The hard link that names the new ledger fails if anything is there, a link that leads
        // nowhere included: the check and the creation are one step.
        let ledger = VerifierLedger::new();
        let bytes = ledger.seal()?;
        reopenable(&bytes, limits)?;
        let file = write_new(&place, &bytes, &lock.held, hooks)?;
        Ok(Self {
            place,
            lock: lock.release(),
            ledger,
            limits: *limits,
            file,
        })
    }

    /// Opens the ledger file at `path` and holds it: the lock in the directory resolved now,
    /// then the file, read as [`read_ledger`] reads it, the path's rules first. A refusal
    /// releases the lock.
    ///
    /// # Errors
    ///
    /// `ERR-RES-IO` when the lock is held or the file cannot be read; and the errors of
    /// [`read_ledger`], `ERR-USAGE` for a path the rules refuse among them.
    pub fn open(path: &Path) -> Result<Self, VerifierError> {
        Self::open_with(path, &LEDGER_LIMITS)
    }

    fn open_with(path: &Path, limits: &DecodeLimits) -> Result<Self, VerifierError> {
        let place = Place::resolve(path)?;
        let lock = Lock::take(&place)?;
        let (ledger, file) = read_ledger_under(&place.file(), limits)?;
        Ok(Self {
            place,
            lock: lock.release(),
            ledger,
            limits: *limits,
            file,
        })
    }

    /// The ledger as last written.
    pub fn ledger(&self) -> &VerifierLedger {
        &self.ledger
    }

    /// The ledger's path: its name in the directory resolved when it was opened.
    pub fn path(&self) -> PathBuf {
        self.place.file()
    }

    /// Records an accepted proof ([`VerifierLedger::record_accepted`]) and writes the ledger.
    ///
    /// # Errors
    ///
    /// Those of [`VerifierLedger::record_accepted`]; `Inconclusive(ERR-RES-IO)` when the file or
    /// its directory cannot be written or synced, when the lock no longer holds this writer's
    /// token, or when the ledger is no longer the file this writer last read or wrote;
    /// `Inconclusive(ERR-USAGE)` when the ledger is no longer one regular file of one name;
    /// `Inconclusive(ERR-CODEC-LIMIT)` when the new ledger would not reopen under
    /// [`LEDGER_LIMITS`]. A batch of no request writes nothing.
    pub fn record_accepted(
        &mut self,
        sample: &AuditSample,
        statement: &PublicStatement,
        uhat: &[FixedBits],
    ) -> Result<(), Failure> {
        self.record_accepted_with(sample, statement, uhat, &REAL)
    }

    fn record_accepted_with(
        &mut self,
        sample: &AuditSample,
        statement: &PublicStatement,
        uhat: &[FixedBits],
        hooks: &Hooks<'_>,
    ) -> Result<(), Failure> {
        let mut next = self.ledger.clone();
        next.record_accepted(sample, statement, uhat)?;
        self.commit(next, hooks)
    }

    /// Records that the verifier derived `sample`, and writes the ledger, before the sample
    /// leaves the verifier (RFC-0067 'The verifier's own sample'). Only the reveal of the
    /// verifier's own toss calls it ([`super::AuditToss::reveal`]). The same sample again writes
    /// nothing; another sample of a manifest the ledger holds is `Inconclusive(ERR-PROTO-ORDER)`
    /// (RFC-0067 'One sample per manifest').
    pub(crate) fn record_derived(&mut self, sample: &AuditSample) -> Result<(), Failure> {
        self.record_derived_with(sample, &REAL)
    }

    fn record_derived_with(
        &mut self,
        sample: &AuditSample,
        hooks: &Hooks<'_>,
    ) -> Result<(), Failure> {
        let mut next = self.ledger.clone();
        next.add_derived(sample)?;
        self.commit(next, hooks)
    }

    /// Writes `next` and makes it the ledger. A ledger that does not change is not written: a
    /// batch of no request, or a sample derived again. One that would not reopen under the
    /// ledger's limits is refused before anything is written.
    fn commit(&mut self, next: VerifierLedger, hooks: &Hooks<'_>) -> Result<(), Failure> {
        if next == self.ledger {
            return Ok(());
        }
        let bytes = next.seal().map_err(Failure::Inconclusive)?;
        reopenable(&bytes, &self.limits).map_err(Failure::Inconclusive)?;
        let guard = Guard {
            lock: &self.lock,
            file: self.file,
        };
        let file =
            write_atomic(&self.place, &bytes, &guard, hooks).map_err(Failure::Inconclusive)?;
        self.file = file;
        self.ledger = next;
        Ok(())
    }
}

/// Whether the sealed ledger `bytes` reopens under `limits`: its size, then the whole decoding,
/// whose sequences and depth the limits bound too. A ledger past them is `ERR-CODEC-LIMIT`.
fn reopenable(bytes: &[u8], limits: &DecodeLimits) -> Result<(), VerifierError> {
    let limit = || VerifierError {
        code: Code::CodecLimit,
    };
    if u64::try_from(bytes.len()).map_or(true, |n| n > limits.max_total_bytes) {
        return Err(limit());
    }
    match VerifierLedger::open(bytes, limits) {
        Ok(_) => Ok(()),
        Err(e) if e.code() == tv_codec::ErrorCode::Limit => Err(limit()),
        Err(e) if e.code() == tv_codec::ErrorCode::Memory => {
            Err(VerifierError { code: Code::Memory })
        }
        // The ledger's own encoding decodes, by INV-TCE-1.
        Err(_) => Err(VerifierError {
            code: Code::Internal,
        }),
    }
}

impl Drop for LedgerFile {
    fn drop(&mut self) {
        self.lock.remove();
    }
}

/// The most bytes a lock file may hold: a token is far shorter.
const LOCK_MAX: u64 = 4096;

/// A lock this writer created: its path, the token it wrote there, and the lock file's identity.
#[derive(Debug)]
struct Held {
    path: PathBuf,
    token: String,
    id: FileId,
}

impl Held {
    /// Whether the lock is still this writer's: the file this writer created, a regular file
    /// read without following a link or blocking, that holds its token. A file's device and inode
    /// can be reused once it is removed, so a lock cleared and created again may have this lock's
    /// identity: the token, unique to this writer, is the half of the check that carries it. The file is checked
    /// twice, by metadata that does not follow a link before it is opened, and by the open file's
    /// own metadata after, so that a lock swapped between the two is not taken for this writer's.
    fn holds(&self) -> bool {
        self.holds_with(&|| ())
    }

    /// [`holds`](Self::holds), with `between` run after the first check and before the open,
    /// where a test swaps the lock.
    fn holds_with(&self, between: &dyn Fn()) -> bool {
        let Ok(meta) = fs::symlink_metadata(&self.path) else {
            return false;
        };
        if !meta.file_type().is_file() || meta.len() > LOCK_MAX || file_id(&meta) != self.id {
            return false;
        }
        between();
        let Ok(mut file) = open_quietly(&self.path) else {
            return false;
        };
        if file
            .metadata()
            .map_or(true, |m| !m.is_file() || file_id(&m) != self.id)
        {
            return false;
        }
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take(LOCK_MAX)
            .read_to_end(&mut bytes)
            .is_ok_and(|_| bytes == self.token.as_bytes())
    }

    /// Removes the lock if it still holds this writer's token (RFC-0067 'One writer'): after an
    /// operator cleared a stale lock, the lock at the path may be another writer's, and it stays.
    ///
    /// The check and the removal are two steps: if, between them, an operator cleared this lock
    /// and another writer took a new one at the path, this writer would remove the new lock. The
    /// window is the few instructions between the read and the unlink; `std` offers no unlink
    /// that names a file by what it holds.
    fn remove(&self) {
        if self.holds() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// The lock file of a ledger while it is being taken: removed if dropped armed.
struct Lock {
    held: Held,
    armed: bool,
}

impl Lock {
    /// Creates `<name>.lock` exclusively and writes this writer's token in it: the process, the
    /// time and a counter, which no other writer's lock repeats. A token that cannot be written
    /// would leave a lock no one could prove theirs, so the take then fails and removes the lock,
    /// if the file at the path is still the one this call created.
    fn take(place: &Place) -> Result<Self, VerifierError> {
        let path = place.beside("", ".lock");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|_| io())?;
        let Ok(id) = file.metadata().map(|m| file_id(&m)) else {
            // The file was created by this call an instant ago and holds no token: it is removed,
            // as nothing could later prove it this writer's.
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(io());
        };
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let token = format!(
            "tessera-verify ledger lock: process {}, {nanos} ns, {}\n",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let written = file
            .write_all(token.as_bytes())
            .and_then(|()| file.sync_all());
        drop(file);
        if written.is_err() {
            if fs::symlink_metadata(&path).is_ok_and(|m| file_id(&m) == id) {
                let _ = fs::remove_file(&path);
            }
            return Err(io());
        }
        Ok(Self {
            held: Held { path, token, id },
            armed: true,
        })
    }

    fn release(mut self) -> Held {
        self.armed = false;
        Held {
            path: core::mem::take(&mut self.held.path),
            token: core::mem::take(&mut self.held.token),
            id: self.held.id,
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        if self.armed {
            self.held.remove();
        }
    }
}

// ----- Writes ----------------------------------------------------------------------------------

/// The steps of a write, at which a test stops it as a crash would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// The temporary file exists, empty.
    Created,
    /// Half the bytes are written.
    Partial,
    /// Every byte is written, not yet synced.
    Written,
    /// The temporary file is synced, not yet renamed.
    Synced,
}

/// The points of a write where a test acts, as another process would: just before the check
/// that this writer still holds the ledger, and just after it, before the rename.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Point {
    /// The temporary file is written and synced; the check comes next.
    Check,
    /// The check has passed; the rename comes next.
    Rename,
}

/// What a test may change in a write: where it stops, what happens around the check, and how
/// the directory is synced.
struct Hooks<'a> {
    stop: &'a dyn Fn(Step) -> bool,
    at: &'a dyn Fn(Point),
    sync_dir: &'a dyn Fn(&Path) -> io::Result<()>,
}

/// The production write: it never stops, nothing happens around the check, and the directory
/// is synced for real.
const REAL: Hooks<'static> = Hooks {
    stop: &never,
    at: &nothing,
    sync_dir: &sync_dir,
};

fn never(_: Step) -> bool {
    false
}

fn nothing(_: Point) {}

/// Syncs a directory, so that a rename or a link in it survives a power loss. On Unix this is
/// `fsync` of the directory; other hosts give `std` no handle on a directory, and have nothing
/// to sync.
fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(())
    }
}

/// What a writer must still hold before it renames: its lock, and the ledger file it last read
/// or wrote.
struct Guard<'a> {
    lock: &'a Held,
    file: FileId,
}

static NEXT: AtomicU64 = AtomicU64::new(0);
/// A temporary file beside the ledger: removed when dropped armed.
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

/// Creates a fresh temporary file in the ledger's directory, readable by its owner only.
fn temporary(place: &Place) -> Result<(File, Temporary), VerifierError> {
    for _ in 0..64 {
        let suffix = format!(
            ".{}.{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let temp = place.beside(".", &suffix);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&temp) {
            Ok(file) => {
                return Ok((
                    file,
                    Temporary {
                        path: temp,
                        armed: true,
                    },
                ));
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(io()),
        }
    }
    Err(io())
}

/// Writes `bytes` over the ledger atomically: a temporary file in its directory, written and
/// synced; then, before the rename, the checks that this writer still holds the ledger (its lock
/// still holds its token, the ledger is still the file it last read or wrote, by device and
/// inode, else `ERR-RES-IO`) and that the ledger is still one regular file of one name (else
/// `ERR-USAGE`); then the rename and the directory's sync. A failure at any step fails the write,
/// and before the rename it writes nothing. Returns the new ledger file's identity.
///
/// The check and the rename are two steps (RFC-0067 'Before each rename'): a writer that passes
/// its check and stalls before the rename, while an operator clears its lock and another writer
/// runs, still renames, over the other writer's record. The window is the few instructions
/// between the check and the rename; closing it needs a lock the kernel holds for the writer,
/// which `std` gives from Rust 1.89, above this workspace's MSRV (#534). The check sits as late
/// as it can, after the temporary file is written and synced.
fn write_atomic(
    place: &Place,
    bytes: &[u8],
    guard: &Guard<'_>,
    hooks: &Hooks<'_>,
) -> Result<FileId, VerifierError> {
    let (mut file, mut temp) = temporary(place)?;
    let crash = |temp: &mut Temporary| {
        temp.armed = false;
        Err(io())
    };
    if (hooks.stop)(Step::Created) {
        return crash(&mut temp);
    }
    let half = bytes.len().checked_div(2).unwrap_or(0);
    let (first, second) = bytes.split_at_checked(half).ok_or(io())?;
    file.write_all(first).map_err(|_| io())?;
    if (hooks.stop)(Step::Partial) {
        return crash(&mut temp);
    }
    file.write_all(second).map_err(|_| io())?;
    if (hooks.stop)(Step::Written) {
        return crash(&mut temp);
    }
    file.sync_all().map_err(|_| io())?;
    let written = file.metadata().map(|m| file_id(&m)).map_err(|_| io())?;
    drop(file);
    if (hooks.stop)(Step::Synced) {
        return crash(&mut temp);
    }
    // Still this writer's: a writer that stalled while an operator cleared its lock, and another
    // writer took the ledger over, would otherwise rename a stale ledger over the other's record.
    (hooks.at)(Point::Check);
    if !guard.lock.holds() {
        return Err(io());
    }
    // A name the ledger gained since it was opened would be a second ledger after the rename.
    let current = checked(&place.file())?;
    if file_id(&current) != guard.file {
        return Err(io());
    }
    (hooks.at)(Point::Rename);
    fs::rename(&temp.path, place.file()).map_err(|_| io())?;
    temp.armed = false;
    (hooks.sync_dir)(&place.dir).map_err(|_| io())?;
    Ok(written)
}

/// Writes `bytes` as a new ledger, where nothing may exist: a synced temporary file,
/// hard-linked to the ledger's name, which fails if anything is there, while the lock still holds
/// this writer's token; then the temporary name removed and the directory synced. Returns the
/// new ledger file's identity.
fn write_new(
    place: &Place,
    bytes: &[u8],
    lock: &Held,
    hooks: &Hooks<'_>,
) -> Result<FileId, VerifierError> {
    let (mut file, mut temp) = temporary(place)?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| io())?;
    let written = file.metadata().map(|m| file_id(&m)).map_err(|_| io())?;
    drop(file);
    if !lock.holds() {
        return Err(io());
    }
    fs::hard_link(&temp.path, place.file()).map_err(|e| {
        if e.kind() == io::ErrorKind::AlreadyExists {
            misuse()
        } else {
            io()
        }
    })?;
    // The ledger must keep one name.
    fs::remove_file(&temp.path).map_err(|_| io())?;
    temp.armed = false;
    (hooks.sync_dir)(&place.dir).map_err(|_| io())?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;
    use crate::audit::ledger::tests::{derived_sample, statement_of};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tessera-ledger-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        fs::canonicalize(dir).unwrap()
    }

    fn leftovers(dir: &Path) -> Vec<PathBuf> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".tmp"))
            .collect()
    }

    #[test]
    fn an_interrupted_write_leaves_the_old_ledger_at_every_step() {
        for step in [Step::Created, Step::Partial, Step::Written, Step::Synced] {
            let dir = scratch("interrupt");
            let path = dir.join("ledger.tvl");
            let mut file = LedgerFile::create(&path).unwrap();
            file.record_derived(&derived_sample(1)).unwrap();
            let old = file.ledger().clone();
            let old_bytes = fs::read(&path).unwrap();
            let stop = move |s: Step| s == step;
            let hooks = Hooks {
                stop: &stop,
                at: &nothing,
                sync_dir: &sync_dir,
            };
            assert_eq!(
                file.record_derived_with(&derived_sample(2), &hooks),
                Err(Failure::Inconclusive(io())),
                "{step:?}"
            );
            // The file is the old ledger, byte for byte, in memory too, and still reads as it.
            assert_eq!(fs::read(&path).unwrap(), old_bytes, "{step:?}");
            assert_eq!(file.ledger(), &old, "{step:?}");
            assert_eq!(read_ledger(&path).unwrap(), old, "{step:?}");
            // The crash left its temporary file, which no reader takes for the ledger.
            assert_eq!(leftovers(&dir).len(), 1, "{step:?}");
            file.record_derived(&derived_sample(2)).unwrap();
            assert_eq!(read_ledger(&path).unwrap().derived().len(), 2);
            drop(file);
            for leftover in leftovers(&dir) {
                fs::remove_file(leftover).unwrap();
            }
            fs::remove_file(&path).unwrap();
            fs::remove_dir(&dir).unwrap();
        }
    }

    #[test]
    fn a_failed_write_removes_its_temporary_file() {
        let dir = scratch("failed");
        let path = dir.join("ledger.tvl");
        let mut file = LedgerFile::create(&path).unwrap();
        // A directory where the ledger should be: the checks refuse it, and nothing is left.
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert_eq!(
            file.record_derived(&derived_sample(1)),
            Err(Failure::Inconclusive(misuse()))
        );
        assert!(leftovers(&dir).is_empty());
        drop(file);
        fs::remove_dir(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn a_failed_directory_sync_fails_the_write_and_changes_nothing_in_memory() {
        let dir = scratch("dirsync");
        let synced: RefCell<Vec<PathBuf>> = RefCell::new(Vec::new());
        let failing = |d: &Path| {
            synced.borrow_mut().push(d.to_path_buf());
            Err(io::Error::other("sync"))
        };
        let hooks = Hooks {
            stop: &never,
            at: &nothing,
            sync_dir: &failing,
        };
        // Creating: the link is made, the directory's sync fails, and so does the creation.
        let path = dir.join("ledger.tvl");
        assert_eq!(
            LedgerFile::create_with(&path, &DecodeLimits::MESSAGE, &hooks).err(),
            Some(io())
        );
        assert_eq!(synced.borrow().as_slice(), std::slice::from_ref(&dir));
        fs::remove_file(&path).unwrap();
        // Recording: the verdict is not Accept, and the ledger in memory is the old one.
        let mut file = LedgerFile::create(&path).unwrap();
        let sample = derived_sample(4);
        file.record_derived(&sample).unwrap();
        let before = file.ledger().clone();
        let statement = statement_of(&sample, &[0]);
        assert_eq!(
            file.record_accepted_with(&sample, &statement, &[FixedBits(1)], &hooks),
            Err(Failure::Inconclusive(io()))
        );
        assert_eq!(file.ledger(), &before);
        assert_eq!(synced.borrow().len(), 2);
        // The rename was done, so the file is no longer the one this writer holds in memory:
        // its later writes are refused before any rename, even with a sync that works (RFC-0067
        // 'A failed directory sync after the rename').
        let bytes = fs::read(&path).unwrap();
        for _ in 0..2 {
            assert_eq!(
                file.record_derived(&derived_sample(5)),
                Err(Failure::Inconclusive(io()))
            );
        }
        assert_eq!(synced.borrow().len(), 2);
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(file.ledger(), &before);
        drop(file);
        // The file holds the record of the write whose sync failed, which a reopened writer
        // reads, and the replay is refused there.
        let mut file = LedgerFile::open(&path).unwrap();
        assert!(file.ledger().proven(sample.identity()).is_some());
        assert_eq!(
            file.record_accepted(&sample, &statement, &[FixedBits(1)]),
            Err(Failure::Rejected(
                crate::verdict::RejectReason::StatementMismatch {
                    field: crate::validation::StatementField::Batch
                }
            ))
        );
        drop(file);
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_ledger_of_two_names_is_refused() {
        let dir = scratch("names");
        let path = dir.join("ledger.tvl");
        drop(LedgerFile::create(&path).unwrap());
        // A symbolic link to it: refused by the reader and the writer, and holds no lock.
        let link = dir.join("link.tvl");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert_eq!(read_ledger(&link).err(), Some(misuse()));
        assert_eq!(LedgerFile::open(&link).err(), Some(misuse()));
        // A ledger is not created over a link, even one that leads nowhere.
        let dangling = dir.join("dangling.tvl");
        std::os::unix::fs::symlink(dir.join("nowhere"), &dangling).unwrap();
        assert_eq!(LedgerFile::create(&dangling).err(), Some(misuse()));
        // A second hard link: both names are refused.
        let hard = dir.join("hard.tvl");
        fs::hard_link(&path, &hard).unwrap();
        for name in [&path, &hard] {
            assert_eq!(LedgerFile::open(name).err(), Some(misuse()));
            assert_eq!(read_ledger(name).err(), Some(misuse()));
        }
        fs::remove_file(&hard).unwrap();
        // A name gained after the open is refused at the next write, which changes nothing.
        let mut file = LedgerFile::open(&path).unwrap();
        fs::hard_link(&path, &hard).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            file.record_derived(&derived_sample(6)),
            Err(Failure::Inconclusive(misuse()))
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(file.ledger(), &VerifierLedger::new());
        drop(file);
        for p in [&hard, &link, &dangling, &path] {
            fs::remove_file(p).unwrap();
        }
        fs::remove_dir(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn the_directory_is_resolved_once_so_two_spellings_share_one_lock() {
        let dir = scratch("spellings");
        let alias = dir.with_extension("alias");
        std::os::unix::fs::symlink(&dir, &alias).unwrap();
        let path = dir.join("ledger.tvl");
        let file = LedgerFile::create(&path).unwrap();
        // The same ledger through a linked directory: the same lock, held.
        let through = alias.join("ledger.tvl");
        assert_eq!(LedgerFile::open(&through).err(), Some(io()));
        assert_eq!(file.path(), path);
        drop(file);
        let file = LedgerFile::open(&through).unwrap();
        assert_eq!(file.path(), path, "the directory as resolved");
        drop(file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&alias).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_is_refused_before_it_is_opened() {
        let dir = scratch("fifo");
        let fifo = dir.join("ledger.tvl");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(made.success());
        // Opening a FIFO for reading blocks until a writer comes: the reader must not open it.
        let (send, receive) = mpsc::channel();
        let path = fifo.clone();
        std::thread::spawn(move || {
            let read = read_ledger(&path).err();
            let open = LedgerFile::open(&path).err();
            let _ = send.send((read, open));
        });
        let (read, open) = receive
            .recv_timeout(Duration::from_secs(20))
            .expect("the FIFO was opened and blocked");
        assert_eq!((read, open), (Some(misuse()), Some(misuse())));
        assert!(!dir.join("ledger.tvl.lock").exists());
        fs::remove_file(&fifo).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn a_reveal_records_its_sample_before_it_is_used() {
        let dir = scratch("derived");
        let path = dir.join("ledger.tvl");
        let mut file = LedgerFile::create(&path).unwrap();
        let sample = derived_sample(7);
        file.record_derived(&sample).unwrap();
        let bytes = fs::read(&path).unwrap();
        // Recorded on disk, and once.
        file.record_derived(&sample).unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        drop(file);
        let ledger = read_ledger(&path).unwrap();
        assert_eq!(
            ledger.derived(),
            [(*sample.manifest_digest(), *sample.identity())]
        );
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn a_lock_taken_over_after_an_operator_cleared_it_survives_the_first_writer() {
        let dir = scratch("takeover");
        let path = dir.join("ledger.tvl");
        let lock = dir.join("ledger.tvl.lock");
        let first = LedgerFile::create(&path).unwrap();
        let token = fs::read(&lock).unwrap();
        assert!(String::from_utf8_lossy(&token).contains(&std::process::id().to_string()));
        // The operator clears what it takes for a stale lock, and a second writer takes over.
        fs::remove_file(&lock).unwrap();
        let second = LedgerFile::open(&path).unwrap();
        let theirs = fs::read(&lock).unwrap();
        assert_ne!(theirs, token, "each writer names itself");
        // The first writer's exit leaves the second's lock; the second's removes it.
        drop(first);
        assert_eq!(fs::read(&lock).unwrap(), theirs);
        assert_eq!(LedgerFile::open(&path).err(), Some(io()));
        drop(second);
        assert!(!lock.exists());
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn an_unchanged_ledger_is_not_written() {
        let dir = scratch("unchanged");
        let path = dir.join("ledger.tvl");
        let mut file = LedgerFile::create(&path).unwrap();
        let sample = derived_sample(3);
        file.record_derived(&sample).unwrap();
        let syncs = RefCell::new(0);
        let counting = |_: &Path| {
            *syncs.borrow_mut() += 1;
            Ok(())
        };
        let hooks = Hooks {
            stop: &never,
            at: &nothing,
            sync_dir: &counting,
        };
        let id = |p: &Path| file_id(&fs::metadata(p).unwrap());
        let before = (id(&path), fs::read(&path).unwrap());
        // A batch of no request, and the same sample derived again: nothing is written.
        file.record_accepted_with(&sample, &statement_of(&sample, &[]), &[], &hooks)
            .unwrap();
        file.record_derived_with(&sample, &hooks).unwrap();
        assert_eq!(*syncs.borrow(), 0);
        assert_eq!((id(&path), fs::read(&path).unwrap()), before);
        assert!(leftovers(&dir).is_empty());
        // A batch of one request is.
        file.record_accepted_with(
            &sample,
            &statement_of(&sample, &[0]),
            &[FixedBits(1)],
            &hooks,
        )
        .unwrap();
        assert_eq!(*syncs.borrow(), 1);
        drop(file);
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn a_write_past_the_reopening_limits_is_refused_and_leaves_the_ledger() {
        let dir = scratch("limits");
        let path = dir.join("ledger.tvl");
        drop(LedgerFile::create(&path).unwrap());
        let size = fs::metadata(&path).unwrap().len();
        // The bytes: room for the empty ledger, not for one more sample of 64 bytes.
        let tight = DecodeLimits {
            max_total_bytes: size + 63,
            ..DecodeLimits::MESSAGE
        };
        let mut file = LedgerFile::open_with(&path, &tight).unwrap();
        let bytes = fs::read(&path).unwrap();
        let limit = Failure::Inconclusive(VerifierError {
            code: Code::CodecLimit,
        });
        assert_eq!(file.record_derived(&derived_sample(1)), Err(limit.clone()));
        assert_eq!(file.ledger(), &VerifierLedger::new());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        drop(file);
        // A sequence's length: one derived sample, not two.
        let one = DecodeLimits {
            max_vec_len: 1,
            ..DecodeLimits::MESSAGE
        };
        let mut file = LedgerFile::open_with(&path, &one).unwrap();
        file.record_derived(&derived_sample(1)).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(file.record_derived(&derived_sample(2)), Err(limit));
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(file.ledger().derived().len(), 1);
        drop(file);
        assert!(
            read_ledger_under(&path, &one).is_ok(),
            "the ledger reopens under its limits"
        );
        // Limits that admit no ledger at all refuse its creation.
        fs::remove_file(&path).unwrap();
        let none = DecodeLimits {
            max_total_bytes: 10,
            ..DecodeLimits::MESSAGE
        };
        assert_eq!(
            LedgerFile::create_with(&path, &none, &REAL).err(),
            Some(VerifierError {
                code: Code::CodecLimit
            })
        );
        assert!(!path.exists());
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn a_file_grown_past_the_limit_after_its_check_is_refused_before_anything_is_reserved() {
        let dir = scratch("grown");
        let path = dir.join("ledger.tvl");
        drop(LedgerFile::create(&path).unwrap());
        let limit = 1 << 20;
        // A sparse petabyte: reserving its size would fail as memory, not as the limit.
        let grow = || {
            let file = OpenOptions::new().write(true).open(&path).unwrap();
            let _ = file.set_len(1 << 50);
        };
        let refused = read_regular_with(&path, limit, &grow).map(|(bytes, _)| bytes);
        assert_eq!(
            refused,
            Err(VerifierError {
                code: Code::CodecLimit
            })
        );
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    /// RFC-0067 'Before each rename', the review's probe: writer A stalls; the operator clears
    /// its lock; writer B opens the ledger and accepts request R; A, whose memory lacks R, accepts
    /// R again. A's rename would wipe B's record and accept R twice (INV-AUD-3). A finds the lock
    /// is not its own, and the ledger not the file it read, and writes nothing.
    #[test]
    fn a_stalled_writer_whose_lock_was_cleared_renames_nothing() {
        let dir = scratch("stalled");
        let path = dir.join("ledger.tvl");
        let lock = dir.join("ledger.tvl.lock");
        let sample = derived_sample(9);
        {
            let mut setup = LedgerFile::create(&path).unwrap();
            setup.record_derived(&sample).unwrap();
        }
        let r = statement_of(&sample, &[1]);
        let mut a = LedgerFile::open(&path).unwrap();
        // The operator takes A's lock for a stale one, and B takes over and accepts R.
        fs::remove_file(&lock).unwrap();
        let mut b = LedgerFile::open(&path).unwrap();
        b.record_accepted(&sample, &r, &[FixedBits(4)]).unwrap();
        let after_b = fs::read(&path).unwrap();
        // A accepts R again from its stale memory: refused, nothing written or left behind.
        assert!(a.ledger().proven(sample.identity()).is_none());
        assert_eq!(
            a.record_accepted(&sample, &r, &[FixedBits(4)]),
            Err(Failure::Inconclusive(io()))
        );
        assert_eq!(fs::read(&path).unwrap(), after_b);
        assert!(leftovers(&dir).is_empty());
        // B's record stands, B still writes, and B's lock outlives A.
        drop(a);
        assert!(lock.exists());
        b.record_accepted(&sample, &statement_of(&sample, &[2]), &[FixedBits(5)])
            .unwrap();
        drop(b);
        let ledger = read_ledger(&path).unwrap();
        assert_eq!(
            ledger
                .proven(sample.identity())
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [(1, FixedBits(4)), (2, FixedBits(5))]
        );
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    /// Each half of the check alone: a lock cleared and not taken again, and a ledger replaced
    /// under a lock that still holds the token, each refuse the write.
    #[test]
    fn a_writer_without_its_lock_or_its_file_writes_nothing() {
        let dir = scratch("halves");
        let path = dir.join("ledger.tvl");
        let lock = dir.join("ledger.tvl.lock");
        let mut file = LedgerFile::create(&path).unwrap();
        // The lock cleared, nobody holding it.
        fs::remove_file(&lock).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            file.record_derived(&derived_sample(1)),
            Err(Failure::Inconclusive(io()))
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
        drop(file);
        // The lock this writer's, the ledger replaced by a copy (another inode, same bytes).
        let mut file = LedgerFile::open(&path).unwrap();
        let copy = dir.join("copy.tvl");
        fs::write(&copy, &bytes).unwrap();
        fs::rename(&copy, &path).unwrap();
        assert_eq!(
            file.record_derived(&derived_sample(1)),
            Err(Failure::Inconclusive(io()))
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert!(lock.exists());
        drop(file);
        assert!(!lock.exists());
        // A lock whose file was swapped for a link to a copy of the token is not this writer's.
        #[cfg(unix)]
        {
            let mut file = LedgerFile::open(&path).unwrap();
            let token = dir.join("token");
            fs::copy(&lock, &token).unwrap();
            fs::remove_file(&lock).unwrap();
            std::os::unix::fs::symlink(&token, &lock).unwrap();
            assert_eq!(
                file.record_derived(&derived_sample(1)),
                Err(Failure::Inconclusive(io()))
            );
            drop(file);
            assert!(fs::symlink_metadata(&lock).is_ok(), "a link is not removed");
            fs::remove_file(&lock).unwrap();
            fs::remove_file(&token).unwrap();
        }
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn a_snapshot_reader_uses_the_writers_limits() {
        let dir = scratch("snapshot-limits");
        let path = dir.join("ledger.tvl");
        drop(LedgerFile::create(&path).unwrap());
        let small = DecodeLimits {
            max_total_bytes: 3,
            ..DecodeLimits::MESSAGE
        };
        assert_eq!(
            read_ledger_under(&path, &small).map(|_| ()),
            Err(VerifierError {
                code: Code::CodecLimit
            })
        );
        assert!(read_ledger(&path).is_ok());
        assert_eq!(LEDGER_LIMITS, DecodeLimits::MESSAGE);
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    /// A lock is this writer's only while it is the file the writer created and holds its token:
    /// a copy of the token in another file, another token in the same file, and a link to the
    /// moved lock each refuse the write.
    #[test]
    fn a_lock_is_the_writers_only_by_its_file_and_its_token() {
        let dir = scratch("token");
        let path = dir.join("ledger.tvl");
        let lock = dir.join("ledger.tvl.lock");
        let mut file = LedgerFile::create(&path).unwrap();
        let token = fs::read(&lock).unwrap();
        let refused = |file: &mut LedgerFile| {
            assert_eq!(
                file.record_derived(&derived_sample(1)),
                Err(Failure::Inconclusive(io()))
            );
        };
        // The token copied into another file, renamed over the lock.
        let copy = dir.join("copy.lock");
        fs::write(&copy, &token).unwrap();
        fs::rename(&copy, &lock).unwrap();
        refused(&mut file);
        drop(file);
        assert!(lock.exists(), "a lock not the writer's file stays");
        fs::remove_file(&lock).unwrap();
        // Another token written into the writer's own lock file.
        let mut file = LedgerFile::open(&path).unwrap();
        fs::write(&lock, b"another writer").unwrap();
        refused(&mut file);
        drop(file);
        assert_eq!(fs::read(&lock).unwrap(), b"another writer");
        fs::remove_file(&lock).unwrap();
        // The lock moved aside and a link to it put in its place.
        #[cfg(unix)]
        {
            let mut file = LedgerFile::open(&path).unwrap();
            let aside = dir.join("aside.lock");
            fs::rename(&lock, &aside).unwrap();
            std::os::unix::fs::symlink(&aside, &lock).unwrap();
            refused(&mut file);
            drop(file);
            assert!(fs::symlink_metadata(&lock).is_ok() && aside.exists());
            fs::remove_file(&lock).unwrap();
            fs::remove_file(&aside).unwrap();
        }
        // With its lock intact, the writer writes.
        let mut file = LedgerFile::open(&path).unwrap();
        file.record_derived(&derived_sample(1)).unwrap();
        drop(file);
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    /// A new ledger is linked only while the lock still holds the creator's token.
    #[test]
    fn a_ledger_is_created_only_under_its_creators_lock() {
        let dir = scratch("create-lock");
        let path = dir.join("ledger.tvl");
        let place = Place::resolve(&path).unwrap();
        let lock = Lock::take(&place).unwrap();
        fs::write(dir.join("ledger.tvl.lock"), b"another writer").unwrap();
        let bytes = VerifierLedger::new().seal().unwrap();
        assert_eq!(write_new(&place, &bytes, &lock.held, &REAL), Err(io()));
        assert!(!path.exists());
        assert!(leftovers(&dir).is_empty());
        drop(lock);
        assert!(
            dir.join("ledger.tvl.lock").exists(),
            "not this writer's lock"
        );
        fs::remove_file(dir.join("ledger.tvl.lock")).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    /// The lock checked twice, around its open: a lock swapped between the two checks, either
    /// way, is not this writer's, whichever check sees the swap.
    #[cfg(unix)]
    #[test]
    fn a_lock_swapped_between_its_checks_is_not_the_writers() {
        let dir = scratch("swap");
        let path = dir.join("ledger.tvl");
        let lock = dir.join("ledger.tvl.lock");
        let aside = dir.join("aside.lock");
        let copy = dir.join("copy.lock");
        let file = LedgerFile::create(&path).unwrap();
        let held = &file.lock;
        let token = fs::read(&lock).unwrap();
        assert!(held.holds());
        // Genuine at the first check, a copy of the token at the open: the open's check sees it.
        let to_copy = || {
            fs::rename(&lock, &aside).unwrap();
            fs::write(&copy, &token).unwrap();
            fs::rename(&copy, &lock).unwrap();
        };
        assert!(!held.holds_with(&to_copy));
        // A copy at the first check, genuine again at the open: the first check sees it, and
        // returns before the swap, which is then made by hand.
        let back = || fs::rename(&aside, &lock).unwrap();
        assert!(!held.holds_with(&back));
        if aside.exists() {
            back();
        }
        assert!(held.holds(), "the genuine lock is back");
        // Genuine at the first check, a link to it at the open: the open refuses the link.
        if HAS_NO_FOLLOW {
            let to_link = || {
                fs::rename(&lock, &aside).unwrap();
                std::os::unix::fs::symlink(&aside, &lock).unwrap();
            };
            assert!(!held.holds_with(&to_link));
            // A link at the first check, genuine at the open: the first check refuses the link,
            // before the swap, which is then made by hand.
            let unlink = || {
                fs::remove_file(&lock).unwrap();
                fs::rename(&aside, &lock).unwrap();
            };
            assert!(!held.holds_with(&unlink));
            if aside.exists() {
                unlink();
            }
            assert!(held.holds());
        }
        drop(file);
        assert!(!lock.exists());
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    /// RFC-0067 'Before each rename': the check sits after the temporary file is written and
    /// synced, just before the rename. Another writer that takes the ledger over just before the
    /// check is seen by it, and the write renames nothing; one that takes it over between the
    /// check and the rename is the residual window (#534), and the stale rename happens. The two
    /// cases together pin the check between the two points.
    #[test]
    fn the_check_sits_just_before_the_rename() {
        for (point, refused) in [(Point::Check, true), (Point::Rename, false)] {
            let dir = scratch("check-point");
            let path = dir.join("ledger.tvl");
            let lock = dir.join("ledger.tvl.lock");
            let sample = derived_sample(9);
            {
                let mut setup = LedgerFile::create(&path).unwrap();
                setup.record_derived(&sample).unwrap();
            }
            let mut a = LedgerFile::open(&path).unwrap();
            let b_slot: RefCell<Option<LedgerFile>> = RefCell::new(None);
            // At `point` in A's write, the operator clears A's lock, and B takes over and accepts
            // request 2.
            let take_over = |p: Point| {
                if p == point && b_slot.borrow().is_none() {
                    fs::remove_file(&lock).unwrap();
                    let mut b = LedgerFile::open(&path).unwrap();
                    b.record_accepted(&sample, &statement_of(&sample, &[2]), &[FixedBits(5)])
                        .unwrap();
                    *b_slot.borrow_mut() = Some(b);
                }
            };
            let hooks = Hooks {
                stop: &never,
                at: &take_over,
                sync_dir: &sync_dir,
            };
            let result = a.record_accepted_with(
                &sample,
                &statement_of(&sample, &[1]),
                &[FixedBits(4)],
                &hooks,
            );
            let b = b_slot.borrow_mut().take().expect("B took over");
            let on_disk = read_ledger(&path).unwrap();
            let record: Vec<u64> = on_disk
                .proven(sample.identity())
                .map(|r| r.iter().map(|(i, _)| i).collect())
                .unwrap_or_default();
            if refused {
                // Seen by the check: nothing renamed, B's record stands.
                assert_eq!(result, Err(Failure::Inconclusive(io())), "{point:?}");
                assert_eq!(record, [2], "{point:?}");
            } else {
                // The residual window: A's check passed before B ran, and A's stale ledger,
                // without B's record, is renamed over it.
                assert_eq!(result, Ok(()), "{point:?}");
                assert_eq!(record, [1], "{point:?}");
            }
            drop(a);
            drop(b);
            for leftover in leftovers(&dir) {
                fs::remove_file(leftover).unwrap();
            }
            fs::remove_file(&path).unwrap();
            fs::remove_dir(&dir).unwrap();
        }
    }

    /// Each target's flags, as its headers give them (see [`NO_FOLLOW_NO_BLOCK`]).
    #[test]
    fn the_open_flags_are_the_targets_own() {
        #[cfg(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "riscv64")
        ))]
        assert_eq!(NO_FOLLOW_NO_BLOCK, 0o400_000 | 0o4_000);
        #[cfg(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "arm")))]
        assert_eq!(NO_FOLLOW_NO_BLOCK, 0o100_000 | 0o4_000);
        #[cfg(target_vendor = "apple")]
        assert_eq!(NO_FOLLOW_NO_BLOCK, 0x0100 | 0x0004);
        // Where the flags are named, the open refuses a final link.
        if HAS_NO_FOLLOW {
            let dir = scratch("flags");
            let target = dir.join("target");
            fs::write(&target, b"x").unwrap();
            let link = dir.join("link");
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &link).unwrap();
            assert!(open_quietly(&target).is_ok());
            assert!(open_quietly(&link).is_err());
            fs::remove_file(&link).unwrap();
            fs::remove_file(&target).unwrap();
            fs::remove_dir(&dir).unwrap();
        }
    }
}
