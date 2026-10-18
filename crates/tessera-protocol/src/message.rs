// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The messages' schemas and envelopes (RFC-0036 'Versions and envelopes', 03 §5).
//!
//! Each message travels in an RFC-0003 envelope, its schema id and version in the header.
//! Schema version 1 is `tessera-protocol/1`, and `tessera-protocol/2` is schema version 2 of the
//! same schemas, so a reader of these types refuses a version-2 envelope with
//! `ERR-CODEC-VERSION`: the two protocols never mix.

use core::fmt;

use tessera_codec::envelope;
use tessera_codec::schema::{self, Schema};
use tessera_codec::{Decode, DecodeError, DecodeLimits, Encode};

use crate::{
    ColumnQueries, NiProof, ProverSeedShare, Round1, Round2Response, Round3Response, SeedCommit,
    SeedReveal,
};

/// A message of the protocol, by its schema (03 §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MessageKind {
    /// [`SeedCommit`], `0x0501`.
    SeedCommit,
    /// [`ProverSeedShare`], `0x0502`.
    ProverSeedShare,
    /// [`SeedReveal`], `0x0503`.
    SeedReveal,
    /// [`Round1`], `0x0510`.
    Round1,
    /// [`Round2Response`], `0x0511`.
    Round2Response,
    /// [`ColumnQueries`], `0x0512`.
    ColumnQueries,
    /// [`Round3Response`], `0x0513`.
    Round3Response,
    /// [`NiProof`], `0x0520`.
    NiProof,
}

impl MessageKind {
    /// Every message, in the order of 03 §5.
    pub const ALL: [Self; 8] = [
        Self::SeedCommit,
        Self::ProverSeedShare,
        Self::SeedReveal,
        Self::Round1,
        Self::Round2Response,
        Self::ColumnQueries,
        Self::Round3Response,
        Self::NiProof,
    ];

    /// The message's schema in the catalog: its id, and version 1, `tessera-protocol/1`.
    pub const fn schema(self) -> Schema {
        match self {
            Self::SeedCommit => schema::SEED_COMMIT,
            Self::ProverSeedShare => schema::PROVER_SEED_SHARE,
            Self::SeedReveal => schema::SEED_REVEAL,
            Self::Round1 => schema::ROUND1,
            Self::Round2Response => schema::ROUND2_RESPONSE,
            Self::ColumnQueries => schema::COLUMN_QUERIES,
            Self::Round3Response => schema::ROUND3_RESPONSE,
            Self::NiProof => schema::NI_PROOF,
        }
    }
}

/// The message's name, as 03 §5 writes it.
impl fmt::Display for MessageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.schema().name)
    }
}

/// A message of `tessera-protocol/1` in its RFC-0003 envelope.
pub trait Message: Encode + Decode {
    /// The message's schema.
    const KIND: MessageKind;

    /// The message's envelope, at its schema's version.
    fn seal(&self) -> Vec<u8> {
        envelope::seal(&Self::KIND.schema(), self)
    }

    /// The message an envelope of its schema holds, checked in RFC-0003's order: the header,
    /// the length, the digest, then the payload, which it must fill. Errors carry offsets from
    /// the start of the envelope.
    fn open(bytes: &[u8], limits: DecodeLimits) -> Result<Self, DecodeError> {
        envelope::open(bytes, &Self::KIND.schema(), limits)
    }
}

impl Message for SeedCommit {
    const KIND: MessageKind = MessageKind::SeedCommit;
}

impl Message for ProverSeedShare {
    const KIND: MessageKind = MessageKind::ProverSeedShare;
}

impl Message for SeedReveal {
    const KIND: MessageKind = MessageKind::SeedReveal;
}

impl Message for Round1 {
    const KIND: MessageKind = MessageKind::Round1;
}

impl Message for Round2Response {
    const KIND: MessageKind = MessageKind::Round2Response;
}

impl Message for ColumnQueries {
    const KIND: MessageKind = MessageKind::ColumnQueries;
}

impl Message for Round3Response {
    const KIND: MessageKind = MessageKind::Round3Response;
}

impl Message for NiProof {
    const KIND: MessageKind = MessageKind::NiProof;
}
