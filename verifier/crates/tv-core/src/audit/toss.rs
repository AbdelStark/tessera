// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's side of the coin toss over a manifest (RFC-0016 'Sample derivation' step 2):
//! a fresh share committed once the policy is fixed, revealed only once both periods' logs are
//! verified; and the toss as the verifier keeps it between the two, the `VerifierToss` artifact
//! (schema `0x0702`, `.tvt`, RFC-0067 'The pending toss'), whose file is [`super::TossFile`].

use core::fmt;
use std::path::PathBuf;

use tv_codec::{Cursor, DecodeError, DecodeLimits, Envelope, Expected, decode, size};
use tv_hash::{Digest, Domain, hash};
use zeroize::Zeroizing;

use super::file::LedgerFile;
use super::policy::AuditPolicy;
use super::sample::{AuditSample, CoinTossTranscript};
use super::seal;
use crate::anchor::{CheckedManifest, Failure, Window, usage};
use crate::verdict::{ErrorCode, VerifierError};

/// A share's length: 32 bytes (RFC-0016 'The shares').
const SHARE_LEN: usize = 32;

/// The verifier's toss over one manifest as it keeps it until the reveal (RFC-0067 'The pending
/// toss'): the manifest, the policy fixed before the commitment, the commitment `c_V`, and the
/// share `s_V`. TCE in that order, the share 32 bytes, in RFC-0003's envelope, schema `0x0702`
/// version 1, `.tvt` (03 §5).
///
/// The share is secret-class until the reveal (05 §3). It stays on the heap, cleared when the
/// value is dropped, and nothing prints it: `Debug` shows the manifest, the policy and the
/// commitment only, and no accessor, `Clone` or comparison exposes it. Its encodings, which hold
/// it, come back in buffers cleared on drop. Only [`AuditToss::commit`], [`super::TossFile`] and
/// the decoders construct one, and the decoders only when the commitment opens.
pub struct VerifierToss {
    manifest_digest: Digest,
    policy: AuditPolicy,
    commitment: Digest,
    share: Zeroizing<Vec<u8>>,
}

impl VerifierToss {
    /// The envelope's schema: `0x0702`, version 1 (03 §5).
    pub const SCHEMA: Expected = Expected {
        schema_id: 0x0702,
        version: 1,
    };

    /// Draws a toss over `manifest`, whose signature has passed, under `policy`: 32 bytes fresh
    /// from the operating system's generator, `/dev/urandom`, and their commitment.
    ///
    /// # Errors
    ///
    /// `ERR-RES-IO` when the generator cannot be read, or on a host with none, and
    /// `ERR-RES-MEMORY` when the share cannot be reserved.
    pub(super) fn draw(
        manifest: &CheckedManifest,
        policy: AuditPolicy,
    ) -> Result<Self, VerifierError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(SHARE_LEN)
            .map_err(|_| error(ErrorCode::Memory))?;
        bytes.resize(SHARE_LEN, 0);
        let mut share = Zeroizing::new(bytes);
        fresh(share.as_mut_slice())?;
        let commitment = hash(Domain::CoinCommit, share.as_slice());
        Ok(Self {
            manifest_digest: *manifest.identity(),
            policy,
            commitment,
            share,
        })
    }

    /// The manifest the toss is over.
    pub fn manifest_digest(&self) -> &Digest {
        &self.manifest_digest
    }

    /// The policy fixed before the commitment.
    pub fn policy(&self) -> &AuditPolicy {
        &self.policy
    }

    /// `c_V = H("coin-commit", s_V)`.
    pub fn commitment(&self) -> &Digest {
        &self.commitment
    }

    /// Opens a `.tvt` artifact: RFC-0003's envelope, schema `0x0702` version 1, in its order,
    /// then the payload's fields ([`decode_payload`](Self::decode_payload)'s rules).
    ///
    /// # Errors
    ///
    /// The first failure of RFC-0003's order, `ERR-CODEC-*`, an envelope whose digest does not
    /// match `ERR-CODEC-DIGEST`; then `ERR-PROTO-COINTOSS` for a commitment that is not
    /// `H("coin-commit", share)`; or `ERR-RES-MEMORY`.
    pub fn open(tvt: &[u8], limits: &DecodeLimits) -> Result<Self, VerifierError> {
        let toss = Envelope::open(tvt, Self::SCHEMA, limits, |b| hash(Domain::Envelope, b))
            .and_then(|envelope| envelope.decode_payload(|c, _| Self::fields(c)))
            .map_err(codec)?;
        toss.opens()?;
        Ok(toss)
    }

    /// Decodes a payload alone (RFC-0067 'The pending toss'): the manifest digest, the policy
    /// ([`AuditPolicy`]'s rules), the commitment, then the share, 32 bytes; bytes past them are
    /// `ERR-CODEC-NONCANONICAL`. Once the payload decodes, the commitment must be
    /// `H("coin-commit", share)`.
    ///
    /// # Errors
    ///
    /// The first failed check, `ERR-CODEC-*`, then `ERR-PROTO-COINTOSS`; or `ERR-RES-MEMORY`.
    pub fn decode_payload(payload: &[u8], limits: &DecodeLimits) -> Result<Self, VerifierError> {
        let toss = decode(payload, limits, Self::fields).map_err(codec)?;
        toss.opens()?;
        Ok(toss)
    }

    fn fields(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let manifest_digest = c.digest()?;
            let policy = AuditPolicy::decode(c)?;
            let commitment = c.digest()?;
            // Read onto the heap, cleared on drop, rather than into an array on the stack: a
            // `[u8; 32]` of 32 `u8`s is the same 32 bytes.
            let share = Zeroizing::new(c.array(SHARE_LEN, size::U8, Cursor::u8)?);
            Ok(Self {
                manifest_digest,
                policy,
                commitment,
                share,
            })
        })
    }

    /// The commitment opens: `c_V = H("coin-commit", s_V)`, else `ERR-PROTO-COINTOSS`.
    fn opens(&self) -> Result<(), VerifierError> {
        if self.share.len() == SHARE_LEN && hash(Domain::CoinCommit, &self.share) == self.commitment
        {
            Ok(())
        } else {
            Err(error(ErrorCode::ProtocolCoinToss))
        }
    }

    /// The payload's TCE encoding, in a buffer cleared on drop, since it holds the share.
    ///
    /// # Errors
    ///
    /// `ERR-RES-MEMORY` when it cannot be reserved.
    pub fn encode_payload(&self) -> Result<Zeroizing<Vec<u8>>, VerifierError> {
        let len = self
            .policy
            .encoded_len()
            .and_then(|policy| policy.checked_add(3 * size::DIGEST))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(error(ErrorCode::Memory))?;
        let mut out = Zeroizing::new(Vec::new());
        // Reserved once, exactly: a buffer that grew would leave a copy behind.
        out.try_reserve_exact(len)
            .map_err(|_| error(ErrorCode::Memory))?;
        out.extend_from_slice(&self.manifest_digest);
        self.policy.encode(&mut out);
        out.extend_from_slice(&self.commitment);
        out.extend_from_slice(&self.share);
        Ok(out)
    }

    /// The `.tvt` artifact: the payload in RFC-0003's envelope, schema `0x0702` version 1, in a
    /// buffer cleared on drop.
    ///
    /// # Errors
    ///
    /// `ERR-RES-MEMORY` when it cannot be reserved.
    pub fn seal(&self) -> Result<Zeroizing<Vec<u8>>, VerifierError> {
        // `seal` reserves its output exactly, so the buffer it returns is the one it wrote.
        seal(Self::SCHEMA, &self.encode_payload()?).map(Zeroizing::new)
    }
}

/// Shows the manifest, the policy and the commitment, never the share (05 §3).
impl fmt::Debug for VerifierToss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VerifierToss")
            .field("manifest_digest", &self.manifest_digest)
            .field("policy", &self.policy)
            .field("commitment", &self.commitment)
            .finish_non_exhaustive()
    }
}

/// The verifier's coin toss over one manifest: its policy, fixed first, its fresh share `s_V`
/// and its commitment `c_V = H("coin-commit", s_V)`, a [`VerifierToss`], and what its reveals
/// have bound it to.
///
/// The share stays on the heap, cleared when the toss is dropped, and nothing prints it: no
/// `Debug` or `Clone` exposes it before [`reveal`](Self::reveal) puts it in a transcript. A share
/// the prover could learn before it answers would let it choose `s_P`, and with it the sample.
pub struct AuditToss {
    toss: VerifierToss,
    /// The prover's share the toss was first revealed against, the only one it takes.
    answer: Option<[u8; 32]>,
    /// The ledger the toss was first revealed into, the only one it records in.
    ledger: Option<PathBuf>,
}

/// A toss read back from its file ([`super::TossFile`]), revealed against no share yet.
impl From<VerifierToss> for AuditToss {
    fn from(toss: VerifierToss) -> Self {
        Self {
            toss,
            answer: None,
            ledger: None,
        }
    }
}

/// Shows the toss as [`VerifierToss`] does, its share never, and what it is bound to: the
/// prover's share, which the prover sent, and the ledger's path.
impl fmt::Debug for AuditToss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuditToss")
            .field("toss", &self.toss)
            .field("answer", &self.answer)
            .field("ledger", &self.ledger)
            .finish()
    }
}

impl AuditToss {
    /// Commits to a share for the toss over `manifest`, whose signature has passed, under
    /// `policy`, which the parties fixed before this commitment ('The policy comes first'). The
    /// share is 32 bytes fresh from the operating system's generator, `/dev/urandom`, for this
    /// toss alone. A verifier that may stop before the reveal keeps the toss in a file first,
    /// [`super::TossFile::commit`] (RFC-0067 'The pending toss').
    ///
    /// # Errors
    ///
    /// `Inconclusive(ERR-RES-IO)` when the generator cannot be read, or on a host with none, and
    /// `Inconclusive(ERR-RES-MEMORY)` when the share cannot be reserved.
    pub fn commit(manifest: &CheckedManifest, policy: AuditPolicy) -> Result<Self, Failure> {
        VerifierToss::draw(manifest, policy)
            .map(Self::from)
            .map_err(Failure::Inconclusive)
    }

    /// `c_V`, which the verifier sends the prover.
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
    /// Both bindings live in this value. A toss read back from its file starts with neither, and
    /// the ledger's `derived` is then what holds one sample per manifest.
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
        if window.period().manifest_digest() != self.toss.manifest_digest() {
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
            Some(first) if first != prover_share => {
                return Err(Failure::Inconclusive(error(ErrorCode::ProtocolOrder)));
            }
            Some(_) => {}
            None => self.answer = Some(prover_share),
        }
        let mut verifier_share = [0; SHARE_LEN];
        if self.toss.share.len() != verifier_share.len() {
            return Err(Failure::Inconclusive(error(ErrorCode::Internal)));
        }
        verifier_share.copy_from_slice(&self.toss.share);
        let toss = CoinTossTranscript {
            verifier_commitment: *self.toss.commitment(),
            prover_share,
            verifier_share,
        };
        let sample = AuditSample::derive(window, self.toss.policy().clone(), toss)?;
        ledger.record_derived(&sample)?;
        Ok(sample)
    }
}

/// Fills `out` from the operating system's generator: `/dev/urandom` on Unix, none elsewhere.
fn fresh(out: &mut [u8]) -> Result<(), VerifierError> {
    #[cfg(unix)]
    {
        use std::io::Read;
        let mut file = std::fs::File::open("/dev/urandom").map_err(|_| error(ErrorCode::Io))?;
        file.read_exact(out).map_err(|_| error(ErrorCode::Io))
    }
    #[cfg(not(unix))]
    {
        let _ = out;
        Err(error(ErrorCode::Io))
    }
}

/// A decoding failure as the verifier's error: its code, `ERR-CODEC-*` or `ERR-RES-MEMORY`.
fn codec(e: DecodeError) -> VerifierError {
    VerifierError {
        code: e.code().into(),
    }
}

const fn error(code: ErrorCode) -> VerifierError {
    VerifierError { code }
}
