// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The ledger's file (RFC-0067 'Updates', 'One writer'): one regular file of one name, held by
//! one writer under the kernel's lock on its directory, read without following a link, and
//! written atomically, its directory synced as part of every write.
//!
//! - **The path.** The ledger is a regular file, neither a symbolic link nor a file of several
//!   hard links, else `ERR-USAGE`: two names for one ledger could be written as two ledgers.
//!   Its metadata is read without following a link, so a FIFO or a device is refused before it
//!   is opened; the file is then opened without following a link and without blocking, where the
//!   host has the flags for it, and compared, by device and inode, with the one checked. Its
//!   directory is resolved once, when the ledger is opened, and every later name is built in it.
//! - **One writer.** A writer opens that directory and holds an exclusive lock on it, the
//!   system's advisory lock, `flock(LOCK_EX | LOCK_NB)` on Unix through
//!   [`std::fs::File::try_lock`], from the open until the [`LedgerFile`] is dropped or its
//!   process ends, a crash included, when the system releases it: no operator step clears it.
//!   A second writer's `try_lock` fails at once, and it is refused with `ERR-RES-IO`, without
//!   waiting. The lock belongs to the directory, not to a name in it: no file holds it, so
//!   removing or renaming files in the directory, or renaming the directory, which keeps its
//!   inode, neither releases it nor lets a second writer in; two paths to one directory, a
//!   symbolic link or a second mount, give one lock; and a directory holds one ledger's writer at
//!   a time. A `flock` lock belongs to an open file description, so a second writer in the same
//!   process is refused as one in another process is. A reader of a snapshot ([`read_ledger`])
//!   takes no lock. The lock is advisory and local: a network file system may not honour it, and
//!   a process that writes the ledger without taking it is not kept out (RFC-0067 'Drawbacks').
//! - **A write.** The new ledger goes to a fresh temporary file beside the old one, which is
//!   synced and renamed over it, and then the directory is synced. A failed directory sync fails
//!   the write with `ERR-RES-IO`: after a power loss the old ledger could come back, and with it
//!   a request accepted again (INV-AUD-3). Before the rename, the ledger must still be the file
//!   this writer last read or wrote, by device and inode, else `ERR-RES-IO` and nothing is
//!   written: an operator who replaced it meanwhile left a ledger this writer has not read. The
//!   directory's path must also still lead to the directory this writer locked, by device and
//!   inode, else `ERR-RES-IO`: a directory put in its place, with the ledger moved into it, is
//!   one a second writer can lock, and this writer's names would be built outside its lock. The
//!   ledger must also still be one regular file of one name. The lock, held throughout, keeps
//!   every other writer out, so the check and the rename leave no window for one. The ledger in
//!   memory changes only once the write has succeeded, so a failed write leaves the caller's
//!   verdict short of `Accept`.
//! - **Off Unix**, where `std` gives no handle on a directory, no directory is locked, as none
//!   is synced: the guarantees above are Unix's (RFC-0067 'Off Unix'). On a Unix host whose `std`
//!   cannot lock a directory, `try_lock` fails, and every writer is refused with `ERR-RES-IO`.
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
use crate::claims::ParamSetId;
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
/// file opened is the one checked; returns the bytes and the file's identity. Its size is checked
/// against the limit before the open and after it, early exits: the read itself holds no more
/// than `limit + 1` bytes ([`read_capped`]), whatever the size says.
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
    // The file may have grown since it was checked: its size is checked again, an early exit,
    // since the read below holds no more than `limit + 1` bytes whatever the size says.
    if after.len() > limit {
        return Err(VerifierError {
            code: Code::CodecLimit,
        });
    }
    let out = read_capped(&mut file, after.len(), limit, &|_| ())?;
    Ok((out, file_id(&after)))
}

/// Reads `file`, whose size its metadata gave as `size`, holding at most `limit + 1` bytes:
/// `min(size, limit + 1)` reserved first, then reads of at most `limit + 1` bytes in all, the
/// buffer growing only within that bound. More than `limit` bytes is `ERR-CODEC-LIMIT`. So
/// nothing ever reserves past the limit, on any platform and whatever the file does between its
/// size and its read: the guarantee does not rest on what the host lets a process reserve, and
/// the size checks before the read are early exits. `reserved` is told the buffer's capacity
/// after each reservation, for a test.
fn read_capped(
    file: &mut impl Read,
    size: u64,
    limit: u64,
    reserved: &dyn Fn(usize),
) -> Result<Vec<u8>, VerifierError> {
    let memory = || VerifierError { code: Code::Memory };
    let cap = usize::try_from(limit.saturating_add(1)).unwrap_or(usize::MAX);
    let first = usize::try_from(size).map_or(cap, |n| n.min(cap));
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(first).map_err(|_| memory())?;
    reserved(out.capacity());
    let mut chunk = [0u8; 8192];
    loop {
        let room = cap.saturating_sub(out.len()).min(chunk.len());
        if room == 0 {
            break;
        }
        let n = match file.read(chunk.get_mut(..room).unwrap_or_default()) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(io()),
        };
        let got = chunk.get(..n).ok_or(io())?;
        if out.len().saturating_add(n) > out.capacity() {
            // Grow by as much again as the buffer holds, never past the cap.
            let grow = out.capacity().max(n).min(cap.saturating_sub(out.len()));
            out.try_reserve_exact(grow).map_err(|_| memory())?;
            reserved(out.capacity());
        }
        out.extend_from_slice(got);
    }
    reserved(out.capacity());
    if u64::try_from(out.len()).map_or(true, |n| n > limit) {
        return Err(VerifierError {
            code: Code::CodecLimit,
        });
    }
    Ok(out)
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

/// The verifier's ledger file, held for writing (RFC-0067 'The file', 'One writer'): the
/// kernel's exclusive lock on its directory keeps every other writer out, in this process or
/// another, until this value is dropped or its process ends, so that two sessions cannot both
/// accept one request (INV-AUD-3). Every update is written atomically, the directory synced, and
/// only while the ledger is still the file this writer last read or wrote, in the directory it
/// locked; the ledger in memory changes only once the write succeeds.
#[derive(Debug)]
pub struct LedgerFile {
    place: Place,
    /// Held from the open to the drop: dropping it releases the lock.
    lock: DirLock,
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
    /// another writer holds the directory, the directory cannot be resolved, opened or locked, or
    /// a file cannot be written or synced; `ERR-RES-MEMORY`.
    pub fn create(path: &Path) -> Result<Self, VerifierError> {
        Self::create_with(path, &LEDGER_LIMITS, &REAL)
    }

    fn create_with(
        path: &Path,
        limits: &DecodeLimits,
        hooks: &Hooks<'_>,
    ) -> Result<Self, VerifierError> {
        let place = Place::resolve(path)?;
        let lock = DirLock::take(&place.dir)?;
        // The hard link that names the new ledger fails if anything is there, a link that leads
        // nowhere included: the check and the creation are one step.
        let ledger = VerifierLedger::new();
        let bytes = ledger.seal()?;
        reopenable(&bytes, limits)?;
        let file = write_new(&place, &bytes, hooks)?;
        Ok(Self {
            place,
            lock,
            ledger,
            limits: *limits,
            file,
        })
    }

    /// Opens the ledger file at `path` and holds it: the lock on the directory resolved now,
    /// then the file, read as [`read_ledger`] reads it, the path's rules first. A refusal
    /// releases the lock.
    ///
    /// # Errors
    ///
    /// `ERR-RES-IO` when another writer holds the directory, the directory cannot be opened or
    /// locked, or the file cannot be read; and the errors of [`read_ledger`], `ERR-USAGE` for a
    /// path the rules refuse among them.
    pub fn open(path: &Path) -> Result<Self, VerifierError> {
        Self::open_with(path, &LEDGER_LIMITS)
    }

    fn open_with(path: &Path, limits: &DecodeLimits) -> Result<Self, VerifierError> {
        let place = Place::resolve(path)?;
        let lock = DirLock::take(&place.dir)?;
        let (ledger, file) = read_ledger_under(&place.file(), limits)?;
        Ok(Self {
            place,
            lock,
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

    /// Records an accepted proof of `sample`, whose claim list names the parameter set `params`
    /// ([`VerifierLedger::record_accepted`]), and writes the ledger. The accounting checks run
    /// first, the parameter set's among them (RFC-0016 'The parameter set'), and a proof they
    /// refuse writes nothing.
    ///
    /// # Errors
    ///
    /// Those of [`VerifierLedger::record_accepted`]; `Inconclusive(ERR-RES-IO)` when the file or
    /// its directory cannot be written or synced, when the ledger is no longer the file this
    /// writer last read or wrote, or when its directory's path no longer leads to the directory
    /// this writer locked;
    /// `Inconclusive(ERR-USAGE)` when the ledger is no longer one regular file of one name;
    /// `Inconclusive(ERR-CODEC-LIMIT)` when the new ledger would not reopen under
    /// [`LEDGER_LIMITS`]. A batch of no request writes nothing.
    pub fn record_accepted(
        &mut self,
        sample: &AuditSample,
        params: &ParamSetId,
        statement: &PublicStatement,
        uhat: &[FixedBits],
    ) -> Result<(), Failure> {
        self.record_accepted_with(sample, params, statement, uhat, &REAL)
    }

    fn record_accepted_with(
        &mut self,
        sample: &AuditSample,
        params: &ParamSetId,
        statement: &PublicStatement,
        uhat: &[FixedBits],
        hooks: &Hooks<'_>,
    ) -> Result<(), Failure> {
        let mut next = self.ledger.clone();
        next.record_accepted(sample, params, statement, uhat)?;
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

// ----- The lock --------------------------------------------------------------------------------

/// The kernel's exclusive lock on a ledger's directory (RFC-0067 'One writer'), held until this
/// value is dropped, which closes the directory and so releases the lock, or until the process
/// ends, when the system releases it. No file names it: nothing in the directory can be removed
/// or renamed to release it, and there is no stale lock for an operator to clear.
///
/// The lock lasts as long as any descriptor of its open file description. `std` opens the
/// directory close-on-exec, so a child process, which starts with a copy of every descriptor,
/// holds the lock only until it executes its program: a lock dropped while another thread spawns
/// a process can outlast its drop by that long, and the next writer is refused meanwhile.
#[derive(Debug)]
struct DirLock {
    /// The directory, open and locked. Off Unix there is none.
    #[cfg(unix)]
    _held: File,
    /// The locked directory's identity, read from the open directory itself.
    id: FileId,
}

impl DirLock {
    /// Opens the directory `dir`, the ledger's as resolved once, and locks it exclusively,
    /// without waiting: `File::try_lock`, `flock(LOCK_EX | LOCK_NB)`. The lock belongs to the
    /// open file description, so another open of the directory conflicts with it, in this process
    /// or another; the directory is checked as the ledger is, so that nothing but a directory is
    /// opened, and nothing blocks the open where the host has the flags for it
    /// ([`NO_FOLLOW_NO_BLOCK`]).
    ///
    /// # Errors
    ///
    /// `ERR-RES-IO` when `dir` is not a directory, or was replaced between its check and its
    /// open; when it cannot be opened; and when it cannot be locked: another writer holds it, or
    /// the host cannot lock a directory.
    #[cfg(unix)]
    fn take(dir: &Path) -> Result<Self, VerifierError> {
        let before = fs::symlink_metadata(dir).map_err(|_| io())?;
        if !before.is_dir() {
            return Err(io());
        }
        let held = open_quietly(dir).map_err(|_| io())?;
        let after = held.metadata().map_err(|_| io())?;
        if !after.is_dir() || file_id(&after) != file_id(&before) {
            return Err(io());
        }
        // A second writer does not wait: its `try_lock` fails, and it is refused (RFC-0067 'The
        // lock').
        held.try_lock().map_err(|_| io())?;
        Ok(Self {
            _held: held,
            id: file_id(&after),
        })
    }

    /// Off Unix `std` gives no handle on a directory: no lock is taken, as no directory is
    /// synced, and the guarantees are Unix's (RFC-0067 'Off Unix').
    #[cfg(not(unix))]
    fn take(dir: &Path) -> Result<Self, VerifierError> {
        let _ = dir;
        Ok(Self { id: (0, 0) })
    }

    /// Whether `dir`, the path every name of the ledger is built in, still leads to the locked
    /// directory, by device and inode: a directory put in its place is outside the lock.
    fn covers(&self, dir: &Path) -> bool {
        fs::metadata(dir).is_ok_and(|m| m.is_dir() && file_id(&m) == self.id)
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

/// The points of a write where a test acts, as an operator or another process would: just before
/// the check that the ledger is still this writer's, and just after it, before the rename.
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

/// What must still hold before a writer renames: its directory's path leads to the directory it
/// locked, and the ledger is the file it last read or wrote.
struct Guard<'a> {
    lock: &'a DirLock,
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
/// synced; then, before the rename, the checks that the ledger is still this writer's (its
/// directory's path leads to the directory it locked, and the ledger is the file it last read or
/// wrote, each by device and inode, else `ERR-RES-IO`) and that the ledger is still one regular
/// file of one name (else `ERR-USAGE`); then the rename and the directory's sync. A failure at
/// any step fails the write, and before the rename it writes nothing. Returns the new ledger
/// file's identity.
///
/// The check and the rename are two steps (RFC-0067 'Before each rename'), but no other writer
/// can run between them: the directory's lock, held from the open, keeps every other writer out
/// of the directory, whatever files an operator removes or renames there meanwhile, and when the
/// directory itself is renamed. A second writer reaches the ledger only through another
/// directory: one put at the directory's path, with the ledger moved into it, which the check
/// refuses before the rename; or, between the check and the rename, with this writer's
/// temporary file moved into it too, the one case the check cannot see. The check sits as late
/// as it can, after the temporary file is written and synced, so that it sees a ledger the
/// operator replaced while the temporary file was written.
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
    // Still this writer's: the names below are built in the directory it locked, which another
    // directory put at its path, with the ledger moved into it, is not.
    (hooks.at)(Point::Check);
    if !guard.lock.covers(&place.dir) {
        return Err(io());
    }
    // A name the ledger gained since it was opened would be a second ledger after the rename.
    let current = checked(&place.file())?;
    // A ledger the operator replaced is one this writer has not read: its records would be lost.
    if file_id(&current) != guard.file {
        return Err(io());
    }
    (hooks.at)(Point::Rename);
    fs::rename(&temp.path, place.file()).map_err(|_| io())?;
    temp.armed = false;
    (hooks.sync_dir)(&place.dir).map_err(|_| io())?;
    Ok(written)
}

/// Writes `bytes` as a new ledger, where nothing may exist, under the directory's lock its caller
/// holds: a synced temporary file, hard-linked to the ledger's name, which fails if anything is
/// there; then the temporary name removed and the directory synced. Returns the new ledger file's
/// identity.
fn write_new(place: &Place, bytes: &[u8], hooks: &Hooks<'_>) -> Result<FileId, VerifierError> {
    let (mut file, mut temp) = temporary(place)?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| io())?;
    let written = file.metadata().map(|m| file_id(&m)).map_err(|_| io())?;
    drop(file);
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
    //! No test here spawns a process. A child holds a copy of every open file description of
    //! its parent until it executes its program, a directory's lock among them, so that a writer
    //! one test drops would hold its lock a while longer, against the next writer of that test.
    //! The tests that spawn a process, a FIFO's maker or a writer of its own, are in
    //! `tests/ledger_processes.rs`, a test binary of their own, one at a time.

    use std::cell::{Cell, RefCell};
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
            file.record_accepted_with(
                &sample,
                sample.policy().params(),
                &statement,
                &[FixedBits(1)],
                &hooks
            ),
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
            file.record_accepted(
                &sample,
                sample.policy().params(),
                &statement,
                &[FixedBits(1)]
            ),
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

    /// RFC-0067 'One writer per directory': two paths to one directory, here a symbolic link to
    /// it and a path through `..`, give one lock, held by whichever writer opened first, by
    /// either path.
    #[cfg(unix)]
    #[test]
    fn the_directory_is_resolved_once_so_two_spellings_share_one_lock() {
        let dir = scratch("spellings");
        let alias = dir.with_extension("alias");
        std::os::unix::fs::symlink(&dir, &alias).unwrap();
        let path = dir.join("ledger.tvl");
        let file = LedgerFile::create(&path).unwrap();
        // The same ledger through a linked directory, or through `..`: the same lock, held, while
        // a reader still reads the ledger by those paths.
        let through = alias.join("ledger.tvl");
        let dotted = dir
            .join("..")
            .join(dir.file_name().unwrap())
            .join("ledger.tvl");
        for spelling in [&through, &dotted] {
            assert!(read_ledger(spelling).is_ok());
            assert_eq!(LedgerFile::open(spelling).err(), Some(io()));
            let other = spelling.with_file_name("other.tvl");
            assert_eq!(LedgerFile::create(&other).err(), Some(io()));
            assert!(!other.exists());
        }
        assert_eq!(file.path(), path);
        drop(file);
        // Taken through the link, the lock keeps out a writer by the directory's own path.
        let file = LedgerFile::open(&through).unwrap();
        assert_eq!(file.path(), path, "the directory as resolved");
        assert_eq!(LedgerFile::open(&path).err(), Some(io()));
        drop(file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&alias).unwrap();
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

    /// RFC-0067 'The lock', 'One writer per directory', 'No stale lock': while a writer holds a
    /// directory, a second `LedgerFile` in this process, whose open of the directory is an open
    /// file description of its own, is refused with `ERR-RES-IO` at once, for the same ledger and
    /// for another in the directory, while a reader still reads; a ledger in another directory is
    /// not. The directory holds no file but the ledger, so there is nothing for an operator to
    /// clear, and the holder's drop lets the next writer in.
    #[cfg(unix)]
    #[test]
    fn one_writer_holds_a_directory_until_it_is_dropped() {
        let dir = scratch("one-writer");
        let path = dir.join("ledger.tvl");
        let other = dir.join("other.tvl");
        drop(LedgerFile::create(&other).unwrap());
        let mut first = LedgerFile::create(&path).unwrap();
        first.record_derived(&derived_sample(1)).unwrap();
        let names = || {
            let mut names: Vec<OsString> = fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name())
                .collect();
            names.sort();
            names
        };
        assert_eq!(names(), ["ledger.tvl", "other.tvl"], "no lock file");
        let started = std::time::Instant::now();
        for _ in 0..3 {
            assert_eq!(LedgerFile::open(&path).err(), Some(io()));
            assert_eq!(LedgerFile::open(&other).err(), Some(io()));
            assert_eq!(LedgerFile::create(&dir.join("third.tvl")).err(), Some(io()));
        }
        assert!(started.elapsed() < Duration::from_secs(10), "no waiting");
        assert!(!dir.join("third.tvl").exists());
        assert_eq!(read_ledger(&path).unwrap(), *first.ledger());
        assert!(read_ledger(&other).is_ok());
        // Another directory has a lock of its own.
        let elsewhere = scratch("one-writer-elsewhere");
        let apart = LedgerFile::create(&elsewhere.join("ledger.tvl")).unwrap();
        // Dropped, the holder lets the next writer in, with no operator step.
        drop(first);
        assert_eq!(names(), ["ledger.tvl", "other.tvl"]);
        let mut second = LedgerFile::open(&path).unwrap();
        second.record_derived(&derived_sample(2)).unwrap();
        assert_eq!(read_ledger(&path).unwrap().derived().len(), 2);
        drop(second);
        drop(LedgerFile::open(&other).unwrap());
        drop(apart);
        for p in [&path, &other, &elsewhere.join("ledger.tvl")] {
            fs::remove_file(p).unwrap();
        }
        fs::remove_dir(&dir).unwrap();
        fs::remove_dir(&elsewhere).unwrap();
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
        file.record_accepted_with(
            &sample,
            sample.policy().params(),
            &statement_of(&sample, &[]),
            &[],
            &hooks,
        )
        .unwrap();
        file.record_derived_with(&sample, &hooks).unwrap();
        assert_eq!(*syncs.borrow(), 0);
        assert_eq!((id(&path), fs::read(&path).unwrap()), before);
        assert!(leftovers(&dir).is_empty());
        // A batch of one request is.
        file.record_accepted_with(
            &sample,
            sample.policy().params(),
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
        // A sparse tebibyte, which ext4 (16 TiB at most), XFS and APFS all allow, and far past what
        // a runner could reserve: reserving its size would fail as memory, not as the limit. A
        // filesystem that refuses the size fails the test here, not by passing it unread.
        let grow = || {
            let file = OpenOptions::new().write(true).open(&path).unwrap();
            file.set_len(1 << 40).unwrap();
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

    /// RFC-0067 'One writer', the window of #534. Writer A passes its check and stalls before
    /// its rename. Meanwhile the operator looks for a lock file to delete, and finds none; renames
    /// the ledger; renames the directory; and removes the ledger. After each, a second writer B
    /// tries to open the ledger by every name it then has, and to create one at a name nothing
    /// holds in the directory: B is refused with `ERR-RES-IO` every time, while a reader still
    /// reads the ledger, so that only the lock refuses B. A resumes, and its record is in the
    /// ledger; once A is dropped, B opens the ledger with no operator step and finds it there.
    #[cfg(unix)]
    #[test]
    fn a_writer_stalled_before_its_rename_loses_no_record() {
        let dir = scratch("window");
        let path = dir.join("ledger.tvl");
        let sample = derived_sample(9);
        {
            let mut setup = LedgerFile::create(&path).unwrap();
            setup.record_derived(&sample).unwrap();
        }
        let mut a = LedgerFile::open(&path).unwrap();
        // B, refused at `at`, which a reader still reads, and at `free`, which holds nothing.
        let refused = |at: Option<&Path>, free: &Path| {
            if let Some(at) = at {
                assert!(read_ledger(at).is_ok(), "{}", at.display());
                assert_eq!(LedgerFile::open(at).err(), Some(io()), "{}", at.display());
            }
            assert!(!free.exists());
            assert_eq!(
                LedgerFile::create(free).err(),
                Some(io()),
                "{}",
                free.display()
            );
            assert!(!free.exists());
        };
        let stalled = Cell::new(0);
        let operator = |p: Point| {
            if p != Point::Rename {
                return;
            }
            stalled.set(stalled.get() + 1);
            // Nothing in the directory but the ledger and A's temporary file: no lock file.
            let mut names: Vec<OsString> = fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name())
                .collect();
            names.retain(|n| n != "ledger.tvl" && !n.to_string_lossy().ends_with(".tmp"));
            assert!(names.is_empty(), "{names:?}");
            assert_eq!(leftovers(&dir).len(), 1, "A's temporary file");
            refused(Some(&path), &dir.join("other.tvl"));
            // The ledger renamed: B is refused by its new name, and at its old one.
            let moved = dir.join("moved.tvl");
            fs::rename(&path, &moved).unwrap();
            refused(Some(&moved), &path);
            fs::rename(&moved, &path).unwrap();
            // The directory renamed: the lock is the directory's, not its name's.
            let renamed = dir.with_extension("renamed");
            fs::rename(&dir, &renamed).unwrap();
            refused(
                Some(&renamed.join("ledger.tvl")),
                &renamed.join("other.tvl"),
            );
            fs::rename(&renamed, &dir).unwrap();
            // The ledger removed: B cannot create one in its place.
            fs::remove_file(&path).unwrap();
            refused(None, &path);
        };
        let hooks = Hooks {
            stop: &never,
            at: &operator,
            sync_dir: &sync_dir,
        };
        assert!(a.ledger().proven(sample.identity()).is_none());
        a.record_accepted_with(
            &sample,
            sample.policy().params(),
            &statement_of(&sample, &[1]),
            &[FixedBits(4)],
            &hooks,
        )
        .unwrap();
        assert_eq!(stalled.get(), 1);
        let record = |ledger: &VerifierLedger| -> Vec<(u64, FixedBits)> {
            ledger
                .proven(sample.identity())
                .map(|r| r.iter().collect())
                .unwrap_or_default()
        };
        // A's record is in the ledger, with the sample it derived, and A still writes.
        let on_disk = read_ledger(&path).unwrap();
        assert_eq!(record(&on_disk), [(1, FixedBits(4))]);
        assert!(on_disk.is_derived(&sample));
        a.record_accepted(
            &sample,
            sample.policy().params(),
            &statement_of(&sample, &[2]),
            &[FixedBits(5)],
        )
        .unwrap();
        drop(a);
        // B gets in now, with no operator step, and finds both records; its replay is refused.
        let mut b = LedgerFile::open(&path).unwrap();
        assert_eq!(record(b.ledger()), [(1, FixedBits(4)), (2, FixedBits(5))]);
        assert_eq!(
            b.record_accepted(
                &sample,
                sample.policy().params(),
                &statement_of(&sample, &[1]),
                &[FixedBits(4)]
            ),
            Err(Failure::Rejected(
                crate::verdict::RejectReason::StatementMismatch {
                    field: crate::validation::StatementField::Batch
                }
            ))
        );
        drop(b);
        assert!(leftovers(&dir).is_empty());
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    /// RFC-0067 'Before each rename': a ledger the operator replaced while the writer held it,
    /// a copy of another inode with the same bytes, is one the writer has not read, and the write
    /// is refused, `ERR-RES-IO`, nothing renamed or left behind.
    #[cfg(unix)]
    #[test]
    fn a_replaced_ledger_is_not_written_over() {
        let dir = scratch("replaced");
        let path = dir.join("ledger.tvl");
        let mut file = LedgerFile::create(&path).unwrap();
        let bytes = fs::read(&path).unwrap();
        let copy = dir.join("copy.tvl");
        fs::write(&copy, &bytes).unwrap();
        fs::rename(&copy, &path).unwrap();
        let replaced = file_id(&fs::metadata(&path).unwrap());
        for _ in 0..2 {
            assert_eq!(
                file.record_derived(&derived_sample(1)),
                Err(Failure::Inconclusive(io()))
            );
        }
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(file_id(&fs::metadata(&path).unwrap()), replaced);
        assert_eq!(file.ledger(), &VerifierLedger::new());
        assert!(leftovers(&dir).is_empty());
        drop(file);
        // Reopened, the writer reads the replacement, and writes it.
        let mut file = LedgerFile::open(&path).unwrap();
        file.record_derived(&derived_sample(1)).unwrap();
        drop(file);
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    /// The directory replaced while writer A holds it: renamed away, a new directory put at its
    /// path and the ledger moved into it, its inode kept, so that the ledger is still the file A
    /// read. Writer B locks the new directory. A's names would now be built outside its lock, and
    /// A, interleaved with B, could rename over B's record: A's writes are refused,
    /// `ERR-RES-IO`, and B is the one writer of the ledger.
    #[cfg(unix)]
    #[test]
    fn a_writer_whose_directory_was_replaced_writes_nothing() {
        let dir = scratch("swapped");
        let path = dir.join("ledger.tvl");
        let sample = derived_sample(9);
        {
            let mut setup = LedgerFile::create(&path).unwrap();
            setup.record_derived(&sample).unwrap();
        }
        let mut a = LedgerFile::open(&path).unwrap();
        let read = file_id(&fs::metadata(&path).unwrap());
        let old = dir.with_extension("old");
        fs::rename(&dir, &old).unwrap();
        fs::create_dir(&dir).unwrap();
        fs::rename(old.join("ledger.tvl"), &path).unwrap();
        assert_eq!(
            file_id(&fs::metadata(&path).unwrap()),
            read,
            "the file A read"
        );
        let mut b = LedgerFile::open(&path).unwrap();
        let bytes = fs::read(&path).unwrap();
        let r = statement_of(&sample, &[1]);
        assert_eq!(
            a.record_accepted(&sample, sample.policy().params(), &r, &[FixedBits(4)]),
            Err(Failure::Inconclusive(io()))
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert!(leftovers(&dir).is_empty() && leftovers(&old).is_empty());
        b.record_accepted(&sample, sample.policy().params(), &r, &[FixedBits(4)])
            .unwrap();
        // A's later writes are refused too, and B's record stands.
        assert_eq!(
            a.record_accepted(
                &sample,
                sample.policy().params(),
                &statement_of(&sample, &[2]),
                &[FixedBits(5)]
            ),
            Err(Failure::Inconclusive(io()))
        );
        drop(a);
        drop(b);
        let ledger = read_ledger(&path).unwrap();
        assert_eq!(
            ledger
                .proven(sample.identity())
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [(1, FixedBits(4))]
        );
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
        fs::remove_dir(&old).unwrap();
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

    /// RFC-0067 'Before each rename': the check sits after the temporary file is written and
    /// synced, just before the rename. A ledger the operator replaces just before the check, a
    /// copy of another inode with the same bytes, is seen by it, and the write renames nothing;
    /// one replaced between the check and the rename is renamed over, the writer's record with
    /// it, which loses nothing, since the lock kept every other writer out and the copy holds
    /// what the writer read. The two cases together pin the check between the two points.
    #[cfg(unix)]
    #[test]
    fn the_check_sits_just_before_the_rename() {
        for (point, refused) in [(Point::Check, true), (Point::Rename, false)] {
            let dir = scratch("check-point");
            let path = dir.join("ledger.tvl");
            let sample = derived_sample(9);
            {
                let mut setup = LedgerFile::create(&path).unwrap();
                setup.record_derived(&sample).unwrap();
            }
            let bytes = fs::read(&path).unwrap();
            let mut a = LedgerFile::open(&path).unwrap();
            let replaced = Cell::new(false);
            let replace = |p: Point| {
                if p == point && !replaced.get() {
                    let copy = dir.join("copy.tvl");
                    fs::write(&copy, &bytes).unwrap();
                    fs::rename(&copy, &path).unwrap();
                    replaced.set(true);
                }
            };
            let hooks = Hooks {
                stop: &never,
                at: &replace,
                sync_dir: &sync_dir,
            };
            let result = a.record_accepted_with(
                &sample,
                sample.policy().params(),
                &statement_of(&sample, &[1]),
                &[FixedBits(4)],
                &hooks,
            );
            assert!(replaced.get(), "{point:?}");
            let record: Vec<u64> = read_ledger(&path)
                .unwrap()
                .proven(sample.identity())
                .map(|r| r.iter().map(|(i, _)| i).collect())
                .unwrap_or_default();
            if refused {
                // Seen by the check: nothing renamed, the copy stands.
                assert_eq!(result, Err(Failure::Inconclusive(io())), "{point:?}");
                assert_eq!(fs::read(&path).unwrap(), bytes, "{point:?}");
                assert!(record.is_empty(), "{point:?}");
                assert!(a.ledger().proven(sample.identity()).is_none());
            } else {
                // After the check: the rename replaces the copy with the writer's ledger.
                assert_eq!(result, Ok(()), "{point:?}");
                assert_eq!(record, [1], "{point:?}");
            }
            assert!(leftovers(&dir).is_empty(), "{point:?}");
            drop(a);
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

    /// The read holds at most `limit + 1` bytes on every platform, however large the size its
    /// metadata gave, and however the file grew after it: the refusal never rests on what the
    /// host lets a process reserve.
    #[test]
    fn a_read_never_reserves_past_the_limit() {
        let limit = 4096u64;
        let most = Cell::new(0usize);
        let seen = |c: usize| most.set(most.get().max(c));
        let content = vec![7u8; 10_000];
        // A size of a tebibyte, as a sparse file's metadata gives it, and more bytes than the
        // limit behind it: refused, with no more than `limit + 1` reserved.
        for size in [1u64 << 40, u64::MAX, 10_000] {
            most.set(0);
            assert_eq!(
                read_capped(&mut &content[..], size, limit, &seen),
                Err(VerifierError {
                    code: Code::CodecLimit
                }),
                "{size}"
            );
            assert!(most.get() <= 4097, "{size}: {} reserved", most.get());
        }
        // A file that grew after its size was read: grown within the bound only, then refused.
        most.set(0);
        assert_eq!(
            read_capped(&mut &content[..], 10, limit, &seen),
            Err(VerifierError {
                code: Code::CodecLimit
            })
        );
        assert!(most.get() <= 4097, "{} reserved", most.get());
        // Growth in steps, as reads of 8 KiB arrive: the buffer stops at the cap, not at the next
        // doubling.
        let large = vec![5u8; 20_000];
        most.set(0);
        assert_eq!(
            read_capped(&mut &large[..], 3000, 10_000, &seen),
            Err(VerifierError {
                code: Code::CodecLimit
            })
        );
        assert!(most.get() <= 10_001, "{} reserved", most.get());
        // An honest file: read whole, with exactly its size reserved, and at the limit exactly.
        for len in [0usize, 100, 4096] {
            most.set(0);
            let bytes = vec![3u8; len];
            let read = read_capped(&mut &bytes[..], len as u64, limit, &seen).unwrap();
            assert_eq!(read, bytes);
            assert_eq!(most.get(), len, "{len}");
        }
        // One byte past the limit is refused.
        let over = vec![1u8; 4097];
        assert_eq!(
            read_capped(&mut &over[..], 4097, limit, &|_| ()),
            Err(VerifierError {
                code: Code::CodecLimit
            })
        );
    }
}
