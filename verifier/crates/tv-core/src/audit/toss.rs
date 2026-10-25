// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's side of the coin toss over a manifest (RFC-0016 'Sample derivation' step 2):
//! a fresh share committed once the policy is fixed, revealed only once both periods' logs are
//! verified.

use std::path::PathBuf;

use tv_hash::{Digest, Domain, hash};
use zeroize::Zeroizing;

use super::file::LedgerFile;
use super::policy::AuditPolicy;
use super::sample::{AuditSample, CoinTossTranscript};
use crate::anchor::{CheckedManifest, Failure, Window, usage};
use crate::verdict::{ErrorCode, VerifierError};

/// The verifier's coin toss over one manifest: its policy, fixed first, its fresh share `s_V`
/// and its commitment `c_V = H("coin-commit", s_V)`.
///
/// The share stays on the heap, cleared when the toss is dropped, and nothing prints it: no
/// `Debug` or `Clone` exposes it before [`reveal`](Self::reveal) puts it in a transcript. A share
/// the prover could learn before it answers would let it choose `s_P`, and with it the sample.
pub struct AuditToss {
    manifest_digest: Digest,
    policy: AuditPolicy,
    share: Zeroizing<Vec<u8>>,
    commitment: Digest,
    /// The prover's share the toss was first revealed against, the only one it takes.
    answer: Option<[u8; 32]>,
    /// The ledger the toss was first revealed into, the only one it records in.
    ledger: Option<PathBuf>,
}

impl AuditToss {
    /// Commits to a share for the toss over `manifest`, whose signature has passed, under
    /// `policy`, which the parties fixed before this commitment ('The policy comes first'). The
    /// share is 32 bytes fresh from the operating system's generator, `/dev/urandom`, for this
    /// toss alone.
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-RES-IO)` when the generator cannot be read, or on a host with none, and
    /// `Inconclusive(ERR-RES-MEMORY)` when the share cannot be reserved.
    pub fn commit(manifest: &CheckedManifest, policy: AuditPolicy) -> Result<Self, Failure> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(32)
            .map_err(|_| fail(ErrorCode::Memory))?;
        bytes.resize(32, 0);
        let mut share = Zeroizing::new(bytes);
        fresh(share.as_mut_slice())?;
        let commitment = hash(Domain::CoinCommit, share.as_slice());
        Ok(Self {
            manifest_digest: *manifest.identity(),
            policy,
            share,
            commitment,
            answer: None,
            ledger: None,
        })
    }

    /// `c_V`, which the verifier sends the prover.
    pub fn commitment(&self) -> &Digest {
        &self.commitment
    }

    /// The manifest the toss is over.
    pub fn manifest_digest(&self) -> &Digest {
        &self.manifest_digest
    }

    /// The policy fixed before the commitment.
    pub fn policy(&self) -> &AuditPolicy {
        &self.policy
    }

    /// Reveals `s_V` against the prover's answer `s_P` and derives the period's sample
    /// ([`AuditSample::derive`]). 'The reveal comes last': `window` holds the period's verified
    /// log and the previous period's, bound by the chain, and its period must be the toss's
    /// manifest. The returned sample's transcript carries `s_V`, which the verifier then sends.
    ///
    /// This is the only way a sample becomes the verifier's own (RFC-0067 'The verifier's own
    /// sample'): `(manifest, sample)` joins `ledger`'s `derived`, and the ledger is written,
    /// before the sample is returned. The accounting and the report refuse every other sample,
    /// however well its coin toss opens.
    ///
    /// One sample per manifest (RFC-0067 'One sample per manifest'): once `s_V` is out, a prover
    /// could choose another share whose sample suits it. So the toss is revealed against one
    /// prover share, the first that reaches the derivation, and any other is
    /// `Inconclusive(ERR-PROTO-ORDER)`; and the ledger takes no second sample for a manifest it
    /// holds, `Inconclusive(ERR-PROTO-ORDER)`, which also refuses a second toss over one
    /// manifest. The same share again gives the same sample, which changes nothing.
    ///
    /// The toss belongs to one ledger, the one it is first revealed into, by its canonical path:
    /// a reveal into another is `Inconclusive(ERR-USAGE)` (RFC-0067 'One ledger per verifier').
    ///
    /// The toss stays usable after a failure, so that a verifier whose inputs disagreed can
    /// reveal once they agree, into the same ledger and against the same share: the prover
    /// answers one commitment per manifest, and a share lost would leave the period without a
    /// sample.
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-USAGE)` for a window over another manifest or another ledger than the
    /// first; `Inconclusive(ERR-PROTO-ORDER)`
    /// for a share other than the first, or a manifest the ledger holds another sample for; the
    /// errors of [`AuditSample::derive`]; and those of the ledger's write, after which the sample
    /// is not returned.
    pub fn reveal(
        &mut self,
        window: &Window<'_>,
        prover_share: [u8; 32],
        ledger: &mut LedgerFile,
    ) -> Result<AuditSample, Failure> {
        if window.period().manifest_digest() != &self.manifest_digest {
            return Err(usage());
        }
        // One ledger per verifier (RFC-0067): a second ledger would hold its own `derived` and
        // `proven`, and could accept a request the first holds.
        let at = ledger.path();
        match &self.ledger {
            Some(first) if first != &at => return Err(usage()),
            Some(_) => {}
            None => self.ledger = Some(at),
        }
        match self.answer {
            Some(first) if first != prover_share => return Err(fail(ErrorCode::ProtocolOrder)),
            Some(_) => {}
            None => self.answer = Some(prover_share),
        }
        let mut verifier_share = [0; 32];
        if self.share.len() != verifier_share.len() {
            return Err(fail(ErrorCode::Internal));
        }
        verifier_share.copy_from_slice(&self.share);
        let toss = CoinTossTranscript {
            verifier_commitment: self.commitment,
            prover_share,
            verifier_share,
        };
        let sample = AuditSample::derive(window, self.policy.clone(), toss)?;
        ledger.record_derived(&sample)?;
        Ok(sample)
    }
}

/// Fills `out` from the operating system's generator: `/dev/urandom` on Unix, none elsewhere.
fn fresh(out: &mut [u8]) -> Result<(), Failure> {
    #[cfg(unix)]
    {
        use std::io::Read;
        let mut file = std::fs::File::open("/dev/urandom").map_err(|_| fail(ErrorCode::Io))?;
        file.read_exact(out).map_err(|_| fail(ErrorCode::Io))
    }
    #[cfg(not(unix))]
    {
        let _ = out;
        Err(fail(ErrorCode::Io))
    }
}

const fn fail(code: ErrorCode) -> Failure {
    Failure::Inconclusive(VerifierError { code })
}
