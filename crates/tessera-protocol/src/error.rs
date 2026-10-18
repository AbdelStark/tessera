// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The one error type of the crate (04 §1): why a session refused a message or cannot go on
//! (RFC-0036 'The prover's live session', 'The verifier's live session', 'Errors'), why a
//! parameter set is not accepted (RFC-0014 'Parameter sets', INV-PROTO-5), and why the soundness
//! calculator cannot bound a statement's sizes (RFC-0014 'Soundness calculator').
//!
//! An error names messages, matrices, counts, positions, lengths and rules, all public (05 §3).
//! None holds a share, a seed, or anything a message carries: a refused query is named by its
//! position, not its index, as the queries' debug form shows counts only.

use core::fmt;

use tessera_codec::error::{ErrorCode, PublicContext, TesseraError};

use crate::{MessageKind, PROTOCOL_VERSION, WeightMatrix};

/// What makes a prover refuse column queries (INV-PROTO-3), checked in this order: the
/// matrices, the counts, then every index's range (`ERR-PROTO-QUERIES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryFault {
    /// The queries name other matrices than `weight_combined`'s, in its order.
    Matrices,
    /// A codeword's queries hold another number of indices than the parameter set's `τ_A` or
    /// `τ_W`.
    Count {
        /// The matrix's id, or `None` for the advice.
        matrix: Option<u8>,
        /// `τ_A` or `τ_W`.
        expected: u32,
        /// The indices the queries hold.
        found: usize,
    },
    /// An index past its codeword.
    Range {
        /// The matrix's id, or `None` for the advice.
        matrix: Option<u8>,
        /// The query's position in draw order, from 0.
        position: usize,
        /// The codeword's length.
        len: u64,
    },
}

/// The advice's or a matrix's codeword, as an error names it.
struct Codeword(Option<u8>);

impl fmt::Display for Codeword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => f.write_str("the advice"),
            Some(id) => match WeightMatrix::from_id(id) {
                Some(matrix) => write!(f, "matrix {matrix}"),
                None => write!(f, "matrix id {id}"),
            },
        }
    }
}

impl fmt::Display for QueryFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Matrices => f.write_str("the queries name other matrices than the opened ones"),
            Self::Count {
                matrix,
                expected,
                found,
            } => write!(
                f,
                "the queries hold {found} indices for {} where the parameter set gives {expected}",
                Codeword(matrix)
            ),
            Self::Range {
                matrix,
                position,
                len,
            } => write!(
                f,
                "query {position} for {} is past its codeword of {len} symbols",
                Codeword(matrix)
            ),
        }
    }
}

/// Why a session refused a message, or cannot go on; why a parameter set is refused; why sizes
/// cannot be bounded.
///
/// - A prover's session refuses a message with [`OutOfOrder`](Self::OutOfOrder),
///   [`OtherSession`](Self::OtherSession), [`Reuse`](Self::Reuse) or [`Queries`](Self::Queries),
///   and is left as it was. [`CoinToss`](Self::CoinToss) aborts it.
/// - A verifier's session ends on any error, `Inconclusive` with its code: a message out of
///   order or of another session, another [`Version`](Self::Version), or a failure of its own.
/// - [`ParamSet::accept`](crate::ParamSet::accept) refuses an identifier with
///   [`UnknownParams`](Self::UnknownParams) or [`TestParams`](Self::TestParams), and
///   [`SoundnessReport::new`](crate::SoundnessReport::new) refuses sizes with
///   [`Sizes`](Self::Sizes): each a configuration the caller chose (`ERR-USAGE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    /// A message the session's state does not expect (`ERR-PROTO-ORDER`).
    OutOfOrder {
        /// The message.
        message: MessageKind,
    },
    /// A message that names another session (`ERR-PROTO-ORDER`).
    OtherSession {
        /// The message.
        message: MessageKind,
    },
    /// A `Round1` of another protocol version than [`PROTOCOL_VERSION`] (`ERR-PROTO-VERSION`).
    Version {
        /// The version it names.
        found: u16,
    },
    /// A `SeedReveal` whose share does not open the verifier's commitment
    /// (`ERR-PROTO-COINTOSS`).
    CoinToss,
    /// A `SeedReveal` after the seed of this `Round1` was answered: a second seed for one
    /// `Round1` (`ERR-PROTO-REUSE`, INV-PROTO-1).
    Reuse,
    /// Column queries INV-PROTO-3 refuses (`ERR-PROTO-QUERIES`).
    Queries(QueryFault),
    /// The operating system's random source failed, as the verifier's session drew its session,
    /// its share or its queries (`ERR-RES-IO`). The prover's session draws nothing.
    Randomness,
    /// The engine or the checks gave the session a value that breaks one of its rules, such as
    /// a codeword the queries cannot index or a query count past `2^16`: a bug on the session's
    /// own side (`ERR-INTERNAL`).
    Internal {
        /// The rule broken.
        rule: &'static str,
    },
    /// An identifier that names none of RFC-0014's parameter sets, such as another version of
    /// one (`ERR-USAGE`).
    UnknownParams,
    /// `test-small/1`, which certifies nothing: only a build with the `test-params` feature
    /// accepts it, and no release build has it (RFC-0001 'Cargo features', `ERR-USAGE`).
    TestParams,
    /// Statement sizes or parameters the soundness calculator cannot bound, such as a codeword
    /// past `2^32` symbols or a count of queries past `2^16` (`ERR-USAGE`).
    Sizes {
        /// The rule broken.
        rule: &'static str,
    },
}

impl ProtocolError {
    /// The stable code (04 §3).
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::OutOfOrder { .. } | Self::OtherSession { .. } => ErrorCode::ProtoOrder,
            Self::Version { .. } => ErrorCode::ProtoVersion,
            Self::CoinToss => ErrorCode::ProtoCointoss,
            Self::Reuse => ErrorCode::ProtoReuse,
            Self::Queries(_) => ErrorCode::ProtoQueries,
            Self::Randomness => ErrorCode::ResIo,
            Self::Internal { .. } => ErrorCode::Internal,
            Self::UnknownParams | Self::TestParams | Self::Sizes { .. } => ErrorCode::Usage,
        }
    }
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfOrder { message } => {
                write!(f, "{message} is out of order for the session's state")
            }
            Self::OtherSession { message } => write!(f, "{message} names another session"),
            Self::Version { found } => write!(
                f,
                "Round1 speaks tessera-protocol/{found}, the session tessera-protocol/{PROTOCOL_VERSION}"
            ),
            Self::CoinToss => {
                f.write_str("the seed reveal does not open the verifier's commitment")
            }
            Self::Reuse => f.write_str("a second seed for one Round1"),
            Self::Queries(fault) => write!(f, "column queries refused: {fault}"),
            Self::Randomness => f.write_str("the operating system's random source failed"),
            Self::Internal { rule } => write!(f, "a rule of the session fails: {rule}"),
            Self::UnknownParams => f.write_str("the identifier names none of RFC-0014's parameter sets"),
            Self::TestParams => f.write_str(
                "test-small/1 certifies nothing: only a build with the test-params feature accepts it",
            ),
            Self::Sizes { rule } => {
                write!(f, "the soundness calculator cannot bound these sizes: {rule}")
            }
        }
    }
}

impl std::error::Error for ProtocolError {}

/// The code, with the schema of the message refused when there is one.
impl From<ProtocolError> for TesseraError {
    fn from(error: ProtocolError) -> Self {
        let message = match error {
            ProtocolError::OutOfOrder { message } | ProtocolError::OtherSession { message } => {
                Some(message)
            }
            ProtocolError::Version { .. } => Some(MessageKind::Round1),
            ProtocolError::CoinToss | ProtocolError::Reuse => Some(MessageKind::SeedReveal),
            ProtocolError::Queries(_) => Some(MessageKind::ColumnQueries),
            ProtocolError::Randomness
            | ProtocolError::Internal { .. }
            | ProtocolError::UnknownParams
            | ProtocolError::TestParams
            | ProtocolError::Sizes { .. } => None,
        };
        TesseraError::new(error.code()).with_context(PublicContext {
            schema: message.map(|message| message.schema().id),
            ..PublicContext::default()
        })
    }
}
