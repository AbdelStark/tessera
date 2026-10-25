// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The pending toss's file (RFC-0067 'The pending toss'): the verifier's toss, kept between its
//! commitment and its reveal in a `.tvt` file of its own, so that a verifier that stops between
//! the two can still reveal. The prover answers one commitment per manifest, so a share lost is
//! the period's audit lost.
//!
//! - **Before the commitment leaves**, [`TossFile::commit`] writes the file as the ledger is
//!   written: a synced temporary file, readable and writable by its owner only (mode 0600 on
//!   Unix), linked into place, which fails if anything is at the path, then the directory synced.
//!   Only the value it returns gives the commitment. A file already at the path is `ERR-USAGE`
//!   and is left as it is: a toss is never silently replaced.
//! - **The reveal.** [`TossFile::open`] reads the file under the ledger's path rules, a regular
//!   file of one name, read without following a link and compared by device and inode with the
//!   file checked, into a buffer cleared on drop; it checks the envelope (`ERR-CODEC-*`, a digest
//!   that does not match `ERR-CODEC-DIGEST`) and the commitment (`ERR-PROTO-COINTOSS`).
//!   [`TossFile::reveal`] is [`AuditToss::reveal`]: the manifest must be the window's
//!   (`ERR-USAGE`), and the sample is recorded in the ledger's `derived` before it is returned.
//! - **Cleared after the record.** [`TossFile::clear`] runs only after a reveal of the same value
//!   has recorded its sample: the file is overwritten with zeros, synced and removed, and its
//!   directory synced. A verifier that stops between the record and the removal opens the file
//!   again and reveals against the same prover share, which gives the same sample and changes
//!   nothing, then clears it.
//! - **Refused, not repaired.** A lost file loses the period's audit: nothing recreates its
//!   share. A file that fails its envelope or does not open its commitment is refused and left
//!   in place, never replaced.

use std::path::{Path, PathBuf};

use tv_codec::DecodeLimits;
use tv_hash::Digest;

use super::file::{self, FileId, LedgerFile, Place};
use super::policy::AuditPolicy;
use super::sample::AuditSample;
use super::toss::{AuditToss, VerifierToss};
use crate::anchor::{CheckedManifest, Failure, Window};
use crate::verdict::{ErrorCode as Code, VerifierError};

/// The decode limits of a pending toss, for its writer and its reader: 1 MiB in all, otherwise
/// `tv-codec`'s message limits. A toss is its policy and 96 bytes, and a policy file the CLI
/// reads, at most 1 MiB of JSON at 28 bytes or more a bucket, gives a policy of fewer than
/// 37,500 buckets of 12 bytes, under 450 KiB. A toss that would not reopen under them is refused
/// before anything is written.
pub const TOSS_LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 1 << 20,
    ..DecodeLimits::MESSAGE
};

/// The verifier's pending toss and its file (RFC-0067 'The pending toss'): created before the
/// commitment leaves, read back after a restart, and cleared once the reveal has recorded its
/// sample in the ledger.
///
/// `Debug` shows what [`AuditToss`]'s does, and the file's place: never the share.
#[derive(Debug)]
pub struct TossFile {
    place: Place,
    /// The toss's file, as created or read.
    file: FileId,
    toss: AuditToss,
    /// Whether a reveal of this value has recorded its sample in a ledger.
    recorded: bool,
}

impl TossFile {
    /// Draws a toss over `manifest`, whose signature has passed, under `policy`, fixed before
    /// the commitment ('The policy comes first'), and writes it to a new file at `path`, before
    /// the commitment leaves: the commitment is read from the value returned, once the file and
    /// its directory are synced.
    ///
    /// # Errors
    ///
    /// `ERR-USAGE` when anything is at `path`, a link that leads nowhere included, which is left
    /// as it is, or when `path` has no file name; `ERR-CODEC-LIMIT` for a toss that would not
    /// reopen under [`TOSS_LIMITS`], before anything is written; `ERR-RES-IO` when the generator
    /// cannot be read, or the directory cannot be resolved, written or synced; `ERR-RES-MEMORY`.
    /// A failure after the file is linked into place, a failed directory sync, leaves a toss
    /// whose commitment never left: refused at the path from then on, it can be removed.
    pub fn commit(
        path: &Path,
        manifest: &CheckedManifest,
        policy: AuditPolicy,
    ) -> Result<Self, VerifierError> {
        let place = Place::resolve(path)?;
        let toss = VerifierToss::draw(manifest, policy)?;
        let bytes = toss.seal()?;
        reopenable(&bytes)?;
        let file = file::create_new(&place, &bytes)?;
        Ok(Self {
            place,
            file,
            toss: AuditToss::from(toss),
            recorded: false,
        })
    }

    /// Reads the pending toss at `path`: a regular file of one name, neither a symbolic link nor
    /// a file of several hard links, read without following a link, then its envelope and its
    /// commitment ([`VerifierToss::open`] under [`TOSS_LIMITS`]).
    ///
    /// # Errors
    ///
    /// `ERR-RES-IO` when nothing can be read at `path`, a lost toss among them; `ERR-USAGE` for
    /// a symbolic link, anything but a regular file, or a file of several names;
    /// `ERR-CODEC-LIMIT` for a file past the limits; the envelope's `ERR-CODEC-*`, a digest that
    /// does not match `ERR-CODEC-DIGEST`; `ERR-PROTO-COINTOSS` for a commitment that is not
    /// `H("coin-commit", share)`; `ERR-RES-MEMORY`. Nothing is written: a refused toss stays as
    /// it is.
    pub fn open(path: &Path) -> Result<Self, VerifierError> {
        let place = Place::resolve(path)?;
        let (bytes, file) = file::read_secret(&place.file(), TOSS_LIMITS.max_total_bytes)?;
        let toss = VerifierToss::open(&bytes, &TOSS_LIMITS)?;
        Ok(Self {
            place,
            file,
            toss: AuditToss::from(toss),
            recorded: false,
        })
    }

    /// `c_V`, which the verifier sends the prover once the file is written.
    pub fn commitment(&self) -> &Digest {
        self.toss.commitment()
    }

    /// The manifest the toss is over.
    pub fn manifest_digest(&self) -> &Digest {
        self.toss.manifest_digest()
    }

    /// The policy fixed before the commitment.
    pub fn policy(&self) -> &AuditPolicy {
        self.toss.policy()
    }

    /// The file's path: its name in the directory resolved when it was created or opened.
    pub fn path(&self) -> PathBuf {
        self.place.file()
    }

    /// Reveals the toss against the prover's share `s_P` and derives the period's sample,
    /// through [`AuditToss::reveal`]: the window's period must be the toss's manifest, and the
    /// sample is recorded in `ledger`'s `derived`, the ledger written, before it is returned.
    /// The file is left as it is; [`clear`](Self::clear) removes it once the sample is safe.
    ///
    /// Revealed again after a stop between the record and the removal, against the same prover
    /// share, the toss gives the same sample, which the ledger holds already: nothing changes.
    /// Against another share, the ledger refuses the other sample, `ERR-PROTO-ORDER`.
    ///
    /// # Errors
    ///
    /// Those of [`AuditToss::reveal`]: `Inconclusive(ERR-USAGE)` for a window over another
    /// manifest, `Inconclusive(ERR-PROTO-ORDER)` for another share, the derivation's and the
    /// ledger write's errors. The toss can then be revealed again.
    pub fn reveal(
        &mut self,
        window: &Window<'_>,
        prover_share: [u8; 32],
        ledger: &mut LedgerFile,
    ) -> Result<AuditSample, Failure> {
        let sample = self.toss.reveal(window, prover_share, ledger)?;
        self.recorded = true;
        Ok(sample)
    }

    /// Clears the toss's file once a reveal of this value has recorded its sample in the ledger
    /// (RFC-0067 'Cleared after the record'): overwritten with zeros, synced, removed, and its
    /// directory synced. Only the file this value created or read is cleared, by device and
    /// inode.
    ///
    /// # Errors
    ///
    /// `ERR-USAGE` before a reveal has recorded the sample, and the file is left as it is;
    /// `ERR-RES-IO` when the file is gone or is no longer the one created or read, or a write, a
    /// sync or the removal fails. A toss whose clearing failed is cleared by opening it again
    /// and revealing it against the same share.
    pub fn clear(self) -> Result<(), VerifierError> {
        if !self.recorded {
            return Err(VerifierError { code: Code::Usage });
        }
        file::clear(&self.place, self.file)
    }
}

/// Whether the sealed toss `bytes` reopens under [`TOSS_LIMITS`]. A toss past them is
/// `ERR-CODEC-LIMIT`; one that does not decode is the writer's fault, by INV-TCE-1.
fn reopenable(bytes: &[u8]) -> Result<(), VerifierError> {
    match VerifierToss::open(bytes, &TOSS_LIMITS) {
        Ok(_) => Ok(()),
        Err(e) if matches!(e.code, Code::CodecLimit | Code::Memory) => Err(e),
        Err(_) => Err(VerifierError {
            code: Code::Internal,
        }),
    }
}
