// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The ledger's file (RFC-0067 'Updates', 'One writer'): one regular file of one name, held by
//! one writer under a lock file, read without following a link, and written atomically, its
//! directory synced as part of every write.
//!
//! - **The path.** The ledger is a regular file, neither a symbolic link nor a file of several
//!   hard links, else `ERR-USAGE`: two names for one ledger would give two locks, and two writers.
//!   Its metadata is read without following a link, so a FIFO or a device is refused before it
//!   is opened; the file opened is then compared, by device and inode, with the one checked. Its
//!   directory is resolved once, when the ledger is opened, and every later name is built in it.
//! - **One writer.** `<name>.lock`, created exclusively in that directory, keeps a second writer
//!   out while the first holds the ledger. A lock left by a crash refuses later writers with
//!   `ERR-RES-IO` until the operator, having checked that no verifier runs, removes it.
//! - **A write.** The new ledger goes to a fresh temporary file beside the old one, which is
//!   synced and renamed over it, and then the directory is synced. A failed directory sync fails
//!   the write with `ERR-RES-IO`: after a power loss the old ledger could come back, and with it
//!   a request accepted again (INV-AUD-3). Before the rename the ledger must still be one regular
//!   file of one name. The ledger in memory changes only once the write has succeeded, so a
//!   failed write leaves the caller's verdict short of `Accept`.

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

const fn io() -> VerifierError {
    VerifierError { code: Code::Io }
}

const fn misuse() -> VerifierError {
    VerifierError { code: Code::Usage }
}

// ----- The path --------------------------------------------------------------------------------

/// What names a file on the host: its device and inode on Unix.
#[cfg(unix)]
fn file_id(meta: &fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (meta.dev(), meta.ino())
}

#[cfg(not(unix))]
fn file_id(_: &fs::Metadata) -> (u64, u64) {
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
/// file opened is the one checked.
fn read_regular(path: &Path, limit: u64) -> Result<Vec<u8>, VerifierError> {
    read_regular_with(path, limit, &|| ())
}

/// [`read_regular`], with `between` run after the check and before the open, where a test makes
/// the file grow.
fn read_regular_with(
    path: &Path,
    limit: u64,
    between: &dyn Fn(),
) -> Result<Vec<u8>, VerifierError> {
    let before = checked(path)?;
    if before.len() > limit {
        return Err(VerifierError {
            code: Code::CodecLimit,
        });
    }
    between();
    let mut file = File::open(path).map_err(|_| io())?;
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
    Ok(out)
}

/// Reads the ledger at `path`, a snapshot for reading only: the file as its last atomic write
/// left it, decoded under `limits`. It takes no lock, since a write replaces the file whole; it
/// applies the path's rules, as every reader of the ledger does.
///
/// # Errors
///
/// `ERR-RES-IO` for a file that cannot be read; `ERR-USAGE` for a symbolic link, a file that is
/// not regular, or one of several hard links; `ERR-CODEC-LIMIT` for a file larger than
/// `limits.max_total_bytes`; and the decoder's errors.
pub fn read_ledger(path: &Path, limits: &DecodeLimits) -> Result<VerifierLedger, VerifierError> {
    let bytes = read_regular(path, limits.max_total_bytes)?;
    VerifierLedger::open(&bytes, limits).map_err(|e| VerifierError {
        code: e.code().into(),
    })
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
/// beside it, `<name>.lock`, created exclusively, keeps a second writer out until this value is
/// dropped, so that two sessions cannot both accept one request (INV-AUD-3). Every update is
/// written atomically, the directory synced, and the ledger in memory changes only once it is.
#[derive(Debug)]
pub struct LedgerFile {
    place: Place,
    lock: Held,
    ledger: VerifierLedger,
    limits: DecodeLimits,
}

impl LedgerFile {
    /// Creates a ledger file at `path`, holding an empty ledger, and holds it. `limits` are the
    /// limits the ledger is to be reopened under: no write makes it larger than they admit.
    ///
    /// # Errors
    ///
    /// `ERR-USAGE` when anything exists at `path`, whose ledger would be lost; `ERR-RES-IO` when
    /// the lock is held, the directory cannot be resolved or a file cannot be written or synced;
    /// `ERR-CODEC-LIMIT` for limits that admit no ledger; `ERR-RES-MEMORY`.
    pub fn create(path: &Path, limits: &DecodeLimits) -> Result<Self, VerifierError> {
        Self::create_with(path, limits, &REAL)
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
        write_new(&place, &bytes, hooks)?;
        Ok(Self {
            place,
            lock: lock.release(),
            ledger,
            limits: *limits,
        })
    }

    /// Opens the ledger file at `path` and holds it: the lock in the directory resolved now,
    /// then the file, read as [`read_ledger`] reads it, the path's rules first. A refusal
    /// releases the lock. `limits` are also the limits no later write may exceed, so that the
    /// ledger always reopens under them.
    ///
    /// # Errors
    ///
    /// `ERR-RES-IO` when the lock is held or the file cannot be read; and the errors of
    /// [`read_ledger`], `ERR-USAGE` for a path the rules refuse among them.
    pub fn open(path: &Path, limits: &DecodeLimits) -> Result<Self, VerifierError> {
        let place = Place::resolve(path)?;
        let lock = Lock::take(&place)?;
        let ledger = read_ledger(&place.file(), limits)?;
        Ok(Self {
            place,
            lock: lock.release(),
            ledger,
            limits: *limits,
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
    /// its directory cannot be written or synced; `Inconclusive(ERR-USAGE)` when the ledger is no
    /// longer one regular file of one name; `Inconclusive(ERR-CODEC-LIMIT)` when the new ledger
    /// would not reopen under the ledger's limits. A batch of no request writes nothing.
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
        write_atomic(&self.place, &bytes, hooks).map_err(Failure::Inconclusive)?;
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

/// A lock this writer created: its path, and the token it wrote there.
#[derive(Debug)]
struct Held {
    path: PathBuf,
    token: String,
}

impl Held {
    /// Removes the lock if it still holds this writer's token (RFC-0067 'One writer'): after an
    /// operator cleared a stale lock, the lock at the path may be another writer's, and it stays.
    fn remove(&self) {
        let ours = fs::symlink_metadata(&self.path)
            .is_ok_and(|m| m.file_type().is_file() && m.len() < 4096)
            && fs::read(&self.path).is_ok_and(|bytes| bytes == self.token.as_bytes());
        if ours {
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
    /// would leave a lock no one could prove theirs, so the lock is then removed and the take
    /// fails.
    fn take(place: &Place) -> Result<Self, VerifierError> {
        let path = place.beside("", ".lock");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|_| io())?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let token = format!(
            "tessera-verify ledger lock: process {}, {nanos} ns, {}\n",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        if file
            .write_all(token.as_bytes())
            .and_then(|()| file.sync_all())
            .is_err()
        {
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(io());
        }
        Ok(Self {
            held: Held { path, token },
            armed: true,
        })
    }

    fn release(mut self) -> Held {
        self.armed = false;
        Held {
            path: core::mem::take(&mut self.held.path),
            token: core::mem::take(&mut self.held.token),
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

/// What a test may change in a write: where it stops, and how the directory is synced.
struct Hooks<'a> {
    stop: &'a dyn Fn(Step) -> bool,
    sync_dir: &'a dyn Fn(&Path) -> io::Result<()>,
}

/// The production write: it never stops, and syncs the directory for real.
const REAL: Hooks<'static> = Hooks {
    stop: &never,
    sync_dir: &sync_dir,
};

fn never(_: Step) -> bool {
    false
}

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

/// Writes `bytes` over the ledger atomically: a temporary file in its directory, written,
/// synced, renamed over the ledger once the ledger is checked to be one regular file of one
/// name, then the directory synced. A failure at any step fails the write.
fn write_atomic(place: &Place, bytes: &[u8], hooks: &Hooks<'_>) -> Result<(), VerifierError> {
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
    drop(file);
    if (hooks.stop)(Step::Synced) {
        return crash(&mut temp);
    }
    // A name the ledger gained since it was opened would be a second ledger after the rename.
    checked(&place.file())?;
    fs::rename(&temp.path, place.file()).map_err(|_| io())?;
    temp.armed = false;
    (hooks.sync_dir)(&place.dir).map_err(|_| io())
}

/// Writes `bytes` as a new ledger, where nothing may exist: a synced temporary file,
/// hard-linked to the ledger's name, which fails if anything is there, the temporary name
/// removed, then the directory synced.
fn write_new(place: &Place, bytes: &[u8], hooks: &Hooks<'_>) -> Result<(), VerifierError> {
    let (mut file, mut temp) = temporary(place)?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| io())?;
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
    (hooks.sync_dir)(&place.dir).map_err(|_| io())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;
    use crate::audit::ledger::tests::{derived_sample, ledger_of, statement_of};

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

    fn place(dir: &Path) -> Place {
        Place::resolve(&dir.join("ledger.tvl")).unwrap()
    }

    #[test]
    fn an_interrupted_write_leaves_the_old_ledger_at_every_step() {
        for step in [Step::Created, Step::Partial, Step::Written, Step::Synced] {
            let dir = scratch("interrupt");
            let at = place(&dir);
            let old = ledger_of(&[&derived_sample(1)]);
            let old_bytes = old.seal().unwrap();
            write_new(&at, &old_bytes, &REAL).unwrap();
            let new_bytes = ledger_of(&[&derived_sample(2)]).seal().unwrap();
            let stop = move |s: Step| s == step;
            let hooks = Hooks {
                stop: &stop,
                sync_dir: &sync_dir,
            };
            assert_eq!(write_atomic(&at, &new_bytes, &hooks), Err(io()), "{step:?}");
            // The file is the old ledger, byte for byte, and still reads as it.
            assert_eq!(fs::read(at.file()).unwrap(), old_bytes, "{step:?}");
            assert_eq!(
                read_ledger(&at.file(), &DecodeLimits::MESSAGE).unwrap(),
                old,
                "{step:?}"
            );
            // The crash left its temporary file, which no reader takes for the ledger.
            assert_eq!(leftovers(&dir).len(), 1, "{step:?}");
            write_atomic(&at, &new_bytes, &REAL).unwrap();
            assert_eq!(fs::read(at.file()).unwrap(), new_bytes);
            for leftover in leftovers(&dir) {
                fs::remove_file(leftover).unwrap();
            }
            fs::remove_file(at.file()).unwrap();
            fs::remove_dir(&dir).unwrap();
        }
    }

    #[test]
    fn a_failed_write_removes_its_temporary_file() {
        let dir = scratch("failed");
        // A directory where the ledger should be: the checks refuse it, and nothing is left.
        let at = place(&dir);
        fs::create_dir(at.file()).unwrap();
        let bytes = ledger_of(&[]).seal().unwrap();
        assert_eq!(write_atomic(&at, &bytes, &REAL), Err(misuse()));
        assert!(leftovers(&dir).is_empty());
        fs::remove_dir(at.file()).unwrap();
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
        let mut file = LedgerFile::create(&path, &DecodeLimits::MESSAGE).unwrap();
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
        assert_eq!(
            file.record_derived_with(&derived_sample(5), &hooks),
            Err(Failure::Inconclusive(io()))
        );
        assert_eq!(file.ledger(), &before);
        // With the real sync, the same record succeeds.
        file.record_accepted(&sample, &statement, &[FixedBits(1)])
            .unwrap();
        drop(file);
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_ledger_of_two_names_is_refused() {
        let dir = scratch("names");
        let path = dir.join("ledger.tvl");
        drop(LedgerFile::create(&path, &DecodeLimits::MESSAGE).unwrap());
        // A symbolic link to it: refused by the reader and the writer, and holds no lock.
        let link = dir.join("link.tvl");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert_eq!(
            read_ledger(&link, &DecodeLimits::MESSAGE).err(),
            Some(misuse())
        );
        assert_eq!(
            LedgerFile::open(&link, &DecodeLimits::MESSAGE).err(),
            Some(misuse())
        );
        // A ledger is not created over a link, even one that leads nowhere.
        let dangling = dir.join("dangling.tvl");
        std::os::unix::fs::symlink(dir.join("nowhere"), &dangling).unwrap();
        assert_eq!(
            LedgerFile::create(&dangling, &DecodeLimits::MESSAGE).err(),
            Some(misuse())
        );
        // A second hard link: both names are refused.
        let hard = dir.join("hard.tvl");
        fs::hard_link(&path, &hard).unwrap();
        for name in [&path, &hard] {
            assert_eq!(
                LedgerFile::open(name, &DecodeLimits::MESSAGE).err(),
                Some(misuse())
            );
            assert_eq!(
                read_ledger(name, &DecodeLimits::MESSAGE).err(),
                Some(misuse())
            );
        }
        fs::remove_file(&hard).unwrap();
        // A name gained after the open is refused at the next write, which changes nothing.
        let mut file = LedgerFile::open(&path, &DecodeLimits::MESSAGE).unwrap();
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
        let file = LedgerFile::create(&path, &DecodeLimits::MESSAGE).unwrap();
        // The same ledger through a linked directory: the same lock, held.
        let through = alias.join("ledger.tvl");
        assert_eq!(
            LedgerFile::open(&through, &DecodeLimits::MESSAGE).err(),
            Some(io())
        );
        assert_eq!(file.path(), path);
        drop(file);
        let file = LedgerFile::open(&through, &DecodeLimits::MESSAGE).unwrap();
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
            let read = read_ledger(&path, &DecodeLimits::MESSAGE).err();
            let open = LedgerFile::open(&path, &DecodeLimits::MESSAGE).err();
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
        let mut file = LedgerFile::create(&path, &DecodeLimits::MESSAGE).unwrap();
        let sample = derived_sample(7);
        file.record_derived(&sample).unwrap();
        let bytes = fs::read(&path).unwrap();
        // Recorded on disk, and once.
        file.record_derived(&sample).unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        drop(file);
        let ledger = read_ledger(&path, &DecodeLimits::MESSAGE).unwrap();
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
        let first = LedgerFile::create(&path, &DecodeLimits::MESSAGE).unwrap();
        let token = fs::read(&lock).unwrap();
        assert!(String::from_utf8_lossy(&token).contains(&std::process::id().to_string()));
        // The operator clears what it takes for a stale lock, and a second writer takes over.
        fs::remove_file(&lock).unwrap();
        let second = LedgerFile::open(&path, &DecodeLimits::MESSAGE).unwrap();
        let theirs = fs::read(&lock).unwrap();
        assert_ne!(theirs, token, "each writer names itself");
        // The first writer's exit leaves the second's lock; the second's removes it.
        drop(first);
        assert_eq!(fs::read(&lock).unwrap(), theirs);
        assert_eq!(
            LedgerFile::open(&path, &DecodeLimits::MESSAGE).err(),
            Some(io())
        );
        drop(second);
        assert!(!lock.exists());
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn an_unchanged_ledger_is_not_written() {
        let dir = scratch("unchanged");
        let path = dir.join("ledger.tvl");
        let mut file = LedgerFile::create(&path, &DecodeLimits::MESSAGE).unwrap();
        let sample = derived_sample(3);
        file.record_derived(&sample).unwrap();
        let syncs = RefCell::new(0);
        let counting = |_: &Path| {
            *syncs.borrow_mut() += 1;
            Ok(())
        };
        let hooks = Hooks {
            stop: &never,
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
        drop(LedgerFile::create(&path, &DecodeLimits::MESSAGE).unwrap());
        let size = fs::metadata(&path).unwrap().len();
        // The bytes: room for the empty ledger, not for one more sample of 64 bytes.
        let tight = DecodeLimits {
            max_total_bytes: size + 63,
            ..DecodeLimits::MESSAGE
        };
        let mut file = LedgerFile::open(&path, &tight).unwrap();
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
        let mut file = LedgerFile::open(&path, &one).unwrap();
        file.record_derived(&derived_sample(1)).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(file.record_derived(&derived_sample(2)), Err(limit));
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(file.ledger().derived().len(), 1);
        drop(file);
        assert!(
            read_ledger(&path, &one).is_ok(),
            "the ledger reopens under its limits"
        );
        // Limits that admit no ledger at all refuse its creation.
        fs::remove_file(&path).unwrap();
        let none = DecodeLimits {
            max_total_bytes: 10,
            ..DecodeLimits::MESSAGE
        };
        assert_eq!(
            LedgerFile::create(&path, &none).err(),
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
        drop(LedgerFile::create(&path, &DecodeLimits::MESSAGE).unwrap());
        let limit = 1 << 20;
        // A sparse petabyte: reserving its size would fail as memory, not as the limit.
        let grow = || {
            let file = OpenOptions::new().write(true).open(&path).unwrap();
            let _ = file.set_len(1 << 50);
        };
        let refused = read_regular_with(&path, limit, &grow);
        assert_eq!(
            refused,
            Err(VerifierError {
                code: Code::CodecLimit
            })
        );
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }
}
