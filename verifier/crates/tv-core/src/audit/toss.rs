// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's side of the coin toss over a manifest (RFC-0016 'Sample derivation' step 2):
//! a fresh share committed once the policy is fixed, revealed only once both periods' logs are
//! verified.

use tv_hash::{Digest, Domain, hash};
use zeroize::Zeroizing;

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
    /// The toss stays usable after a failure, so that a verifier whose inputs disagreed can
    /// reveal once they agree: the prover answers one commitment per manifest, and a share lost
    /// would leave the period without a sample.
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-USAGE)` for a window over another manifest, and the errors of
    /// [`AuditSample::derive`].
    pub fn reveal(
        &self,
        window: &Window<'_>,
        prover_share: [u8; 32],
    ) -> Result<AuditSample, Failure> {
        if window.period().manifest_digest() != &self.manifest_digest {
            return Err(usage());
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
        AuditSample::derive(window, self.policy.clone(), toss)
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
