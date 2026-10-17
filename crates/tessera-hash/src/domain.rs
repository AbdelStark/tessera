// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `H(domain, bytes)`: BLAKE3 in key-derivation mode, with one context string per use
//! (RFC-0004 'Hash function and domains').

use std::sync::LazyLock;

use blake3::hazmat::{ContextKey, HasherExt, hash_derive_key_context};

/// A 32-byte BLAKE3 output (03 §1).
pub type Digest = [u8; 32];

/// Defines [`Domain`], [`Domain::ALL`] and [`Domain::context`] from one list, so that the
/// three cannot disagree: every variant is in `ALL`, in declaration order, which the context-key
/// table relies on.
macro_rules! domains {
    ($($(#[doc = $doc:literal])* $variant:ident => $name:literal,)*) => {
        /// The hash domains of RFC-0004. Distinct context strings make the outputs of different
        /// domains independent.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Domain {
            $($(#[doc = $doc])* $variant,)*
        }

        const DOMAINS: usize = [$($name),*].len();

        impl Domain {
            /// Every domain, in the order of RFC-0004's table, which is also declaration order.
            pub const ALL: [Self; DOMAINS] = [$(Self::$variant),*];

            /// The BLAKE3 context string: `tessera/v1/` followed by the domain's name.
            pub const fn context(self) -> &'static str {
                match self {
                    $(Self::$variant => concat!("tessera/v1/", $name),)*
                }
            }
        }
    };
}

domains! {
    /// Envelope header and payload (RFC-0003).
    Envelope => "envelope",
    /// A commitment's identifier: `TCE(CommitmentDescriptor)` (RFC-0031), from which every
    /// column chain of the commitment starts.
    CommitmentId => "commitment-id",
    /// The first link of a column chain: `commitment_id ‖ column` (RFC-0004 'Column digests').
    ColumnInit => "column-init",
    /// A link of a column chain: `chain ‖ symbols`.
    ColumnStep => "column-step",
    /// A Merkle node: `left ‖ right`.
    MerkleNode => "merkle-node",
    /// An advice leaf from the segment digests of one column (RFC-0011).
    ColumnMerge => "column-merge",
    /// The weight root (RFC-0012).
    WeightsRoot => "weights-root",
    /// The transcript's first state: `TCE(SessionContext)`.
    TranscriptInit => "transcript-init",
    /// A transcript absorb: `state ‖ Ascii(label) ‖ TCE(value)`.
    TranscriptAbsorb => "transcript-absorb",
    /// A transcript squeeze: `state ‖ Ascii(label) ‖ counter`.
    TranscriptSqueeze => "transcript-squeeze",
    /// The transcript state after a squeeze: `state ‖ out`.
    TranscriptRatchet => "transcript-ratchet",
    /// A proof-of-work attempt: `state ‖ Ascii(label) ‖ nonce`.
    Pow => "pow",
    /// The verifier's seed-share commitment (RFC-0014).
    CoinCommit => "coin-commit",
    /// The live seed from both shares (RFC-0014).
    CoinToss => "coin-toss",
    /// A table digest (RFC-0008).
    Table => "table",
    /// The audit sampling seed (RFC-0016).
    AuditSample => "audit-sample",
    /// A per-request audit selection (RFC-0016).
    AuditSelect => "audit-select",
    /// The claim-list digest (RFC-0007).
    ClaimList => "claim-list",
    /// The statement digest a session binds: `TCE(PublicStatement)` (RFC-0007).
    Statement => "statement",
}

/// The context key of every domain, derived once. Deriving it costs one compression, as much as
/// hashing a Merkle node, so doing it per hash would double the cost of short inputs.
static CONTEXT_KEYS: LazyLock<[ContextKey; DOMAINS]> =
    LazyLock::new(|| Domain::ALL.map(|domain| hash_derive_key_context(domain.context())));

/// `H(domain, ·)` over an input given in parts.
#[derive(Clone, Debug)]
pub struct Hasher(blake3::Hasher);

impl Hasher {
    /// An empty input in `domain`.
    pub fn new(domain: Domain) -> Self {
        Self(blake3::Hasher::new_from_context_key(
            &CONTEXT_KEYS[domain as usize],
        ))
    }

    /// Appends `bytes` to the input.
    pub fn update(&mut self, bytes: &[u8]) -> &mut Self {
        self.0.update(bytes);
        self
    }

    /// `H(domain, input)` for the input so far.
    pub fn finalize(&self) -> Digest {
        *self.0.finalize().as_bytes()
    }
}

/// A [`Hasher`] takes TCE encodings directly, so values are hashed as they are encoded, without
/// a buffer.
impl tessera_codec::Sink for Hasher {
    fn put(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}

/// `H(domain, bytes) = BLAKE3.derive_key("tessera/v1/" + domain, bytes)`.
pub fn hash(domain: Domain, bytes: &[u8]) -> Digest {
    Hasher::new(domain).update(bytes).finalize()
}
