// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Tessera hash `H(domain, bytes)` (RFC-0004 'Hash function and domains'): BLAKE3
//! `derive_key` with the context `"tessera/v1/" + domain`, 32-byte output.

use core::fmt;

use crate::Digest;
use crate::blake3::{
    self, BLOCK_LEN, CHUNK_LEN, DERIVE_KEY_MATERIAL, Hasher, context_key, digest_words, pair_block,
};

/// Declares [`Domain`] from one list of variants and names, so that a variant's name, context
/// string and key cannot disagree.
macro_rules! domains {
    ($($(#[doc = $doc:literal])* $variant:ident => $name:literal,)*) => {
        /// A hash domain of RFC-0004 'Hash function and domains'.
        ///
        /// Each domain hashes in BLAKE3's key-derivation mode under its own context string,
        /// `"tessera/v1/"` followed by its name, so that outputs of different domains are
        /// independent. The context keys are derived by the compiler, once.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Domain {
            $($(#[doc = $doc])* $variant,)*
        }

        impl Domain {
            /// Every domain, in the order of RFC-0004's table.
            pub const ALL: [Domain; 18] = [$(Domain::$variant,)*];

            /// The domain's name in RFC-0004's table, such as `"merkle-node"`.
            pub const fn name(self) -> &'static str {
                match self {
                    $(Domain::$variant => $name,)*
                }
            }

            /// The BLAKE3 `derive_key` context string: `"tessera/v1/"` followed by the name.
            pub const fn context(self) -> &'static str {
                match self {
                    $(Domain::$variant => concat!("tessera/v1/", $name),)*
                }
            }

            /// The key BLAKE3 derives from the context string, evaluated at compile time.
            /// [`context_key`] reads one block, so a longer context fails to compile.
            pub(crate) const fn key(self) -> [u32; 8] {
                match self {
                    $(Domain::$variant => const {
                        assert!(concat!("tessera/v1/", $name).len() <= BLOCK_LEN);
                        context_key(concat!("tessera/v1/", $name))
                    },)*
                }
            }
        }
    };
}

domains! {
    /// `envelope`: an artifact envelope's header and payload (RFC-0003 'Envelope').
    Envelope => "envelope",
    /// `column-init`: `commitment_id ‖ u64(column)`, the start of a column's chain.
    ColumnInit => "column-init",
    /// `column-step`: `chain ‖ symbols`, one block of rows of a column.
    ColumnStep => "column-step",
    /// `merkle-node`: `left ‖ right`, a node of a commitment's Merkle tree.
    MerkleNode => "merkle-node",
    /// `column-merge`: the segment digests of one column, in segment order (RFC-0011).
    ColumnMerge => "column-merge",
    /// `weights-root`: `TCE(WeightLayoutParams) ‖ matrix roots ‖ epoch` (RFC-0012).
    WeightsRoot => "weights-root",
    /// `transcript-init`: `TCE(SessionContext)`, the transcript's first state.
    TranscriptInit => "transcript-init",
    /// `transcript-absorb`: `state ‖ Ascii(label) ‖ TCE(value)`.
    TranscriptAbsorb => "transcript-absorb",
    /// `transcript-squeeze`: `state ‖ Ascii(label) ‖ u64(counter)`.
    TranscriptSqueeze => "transcript-squeeze",
    /// `transcript-ratchet`: `state ‖ out`, the state after a squeeze.
    TranscriptRatchet => "transcript-ratchet",
    /// `pow`: `state ‖ Ascii(label) ‖ u64(nonce)`, a proof-of-work attempt.
    Pow => "pow",
    /// `coin-commit`: `share_v`, the verifier's seed-share commitment (RFC-0014).
    CoinCommit => "coin-commit",
    /// `coin-toss`: `share_v ‖ share_p`, the live seed (RFC-0014).
    CoinToss => "coin-toss",
    /// `table`: `TCE(TableSpec without digest) ‖ piece parameters ‖ core values` (RFC-0008).
    Table => "table",
    /// `audit-sample`: the audit sampling seed (RFC-0016).
    AuditSample => "audit-sample",
    /// `audit-select`: the per-request selection inputs of audit sampling (RFC-0016).
    AuditSelect => "audit-select",
    /// `claim-list`: `TCE(ClaimList)` (RFC-0007).
    ClaimList => "claim-list",
    /// `statement`: `TCE(PublicStatement)`, which is `SessionContext.statement_digest` (RFC-0007).
    Statement => "statement",
}

/// `H(domain, input)`: BLAKE3 `derive_key` with the domain's context and `input` as the key
/// material, 32 bytes of output. Every input length is hashed; nothing fails.
pub fn hash(domain: Domain, input: &[u8]) -> Digest {
    if input.len() <= CHUNK_LEN {
        blake3::single_chunk(&domain.key(), DERIVE_KEY_MATERIAL, input)
    } else {
        let mut hasher = Hasher::new(domain.key(), DERIVE_KEY_MATERIAL);
        hasher.update(input);
        hasher.finalize()
    }
}

/// `H(domain, left ‖ right)` for two digests, a single compression.
pub(crate) fn hash_pair(domain: Domain, left: &Digest, right: &Digest) -> Digest {
    let block = pair_block(&digest_words(left), &digest_words(right));
    blake3::single_block(&domain.key(), DERIVE_KEY_MATERIAL, &block)
}

/// `H(domain, ·)` over an input given in pieces: the result is [`hash`] of their concatenation.
///
/// Inputs longer than a few chunks hash fastest when given in large pieces: whole subtrees of
/// the input are then hashed several chunks at a time.
///
/// ```
/// use tv_hash::{Domain, DomainHasher, hash};
///
/// let mut hasher = DomainHasher::new(Domain::ColumnMerge);
/// hasher.update(b"segment one").update(b"segment two");
/// assert_eq!(hasher.finalize(), hash(Domain::ColumnMerge, b"segment onesegment two"));
/// ```
#[derive(Clone)]
pub struct DomainHasher {
    domain: Domain,
    inner: Hasher,
}

impl DomainHasher {
    /// A hasher for `domain` that has seen no input.
    pub fn new(domain: Domain) -> Self {
        Self {
            domain,
            inner: Hasher::new(domain.key(), DERIVE_KEY_MATERIAL),
        }
    }

    /// The domain this hasher hashes under.
    pub fn domain(&self) -> Domain {
        self.domain
    }

    /// Appends `input` to the bytes hashed so far.
    pub fn update(&mut self, input: &[u8]) -> &mut Self {
        self.inner.update(input);
        self
    }

    /// `H(domain, everything appended so far)`. The hasher is unchanged and may take more input.
    pub fn finalize(&self) -> Digest {
        self.inner.finalize()
    }
}

/// Shows the domain, never the input.
impl fmt::Debug for DomainHasher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DomainHasher")
            .field("domain", &self.domain)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blake3::{DERIVE_KEY_CONTEXT, IV};

    #[test]
    fn compile_time_keys_equal_keys_derived_by_the_hasher() {
        for domain in Domain::ALL {
            let context = domain.context();
            assert!(context.len() <= 64, "{context} fits one block");
            assert_eq!(context, format!("tessera/v1/{}", domain.name()));
            let mut hasher = Hasher::new(IV, DERIVE_KEY_CONTEXT);
            hasher.update(context.as_bytes());
            assert_eq!(domain.key(), digest_words(&hasher.finalize()), "{context}");
        }
    }

    #[test]
    fn the_pair_path_is_the_hash_of_the_concatenation() {
        let left = [0x11; 32];
        let right = [0xee; 32];
        let joined = [left, right].concat();
        for domain in Domain::ALL {
            assert_eq!(hash_pair(domain, &left, &right), hash(domain, &joined));
        }
    }
}
