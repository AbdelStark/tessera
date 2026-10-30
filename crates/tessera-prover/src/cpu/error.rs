// SPDX-License-Identifier: Apache-2.0 OR MIT
use super::*;
use core::fmt;
/// Public producer failures; no variant carries a witness, mask, token or seed.
#[derive(Debug)]
pub enum ProverError {
    /// A protocol refusal, preserving RFC-0036 state-machine semantics.
    Protocol(protocol::ProtocolError),
    /// Invalid application configuration or statement.
    Usage {
        /// Public rule violated.
        rule: &'static str,
    },
    /// Model/public compilation failure.
    Compile(tessera_compiler::CompileError),
    /// Integer witness or exact-greedy completeness failure.
    Witness(tessera_witness::WitnessError),
    /// Leaf proof failure.
    Leaf(crate::LeafError),
    /// Stage/native proof failure.
    Stage(crate::StageError),
    /// Commitment/fold failure.
    Pcs(pcs::PcsError),
    /// Canonical row-source failure.
    Advice(crate::advice::Error),
    /// Regenerated opening failure.
    Opening(pcs::AdviceOpeningError),
    /// Resource failure, with its stable public code.
    Resource(ErrorCode),
}
impl ProverError {
    /// Stable code of the failure (04 section 3).
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Protocol(e) => e.code(),
            Self::Usage { .. } => ErrorCode::Usage,
            Self::Compile(e) => e.code(),
            Self::Witness(e) => e.code(),
            Self::Leaf(e) => e.code(),
            Self::Stage(e) => e.code(),
            Self::Opening(e) => e.code(),
            Self::Resource(e) => *e,
            Self::Pcs(pcs::PcsError::TooLarge { .. } | pcs::PcsError::LayoutMemory) => {
                ErrorCode::ResMemory
            }
            Self::Advice(crate::advice::Error::Memory) => ErrorCode::ResMemory,
            Self::Advice(crate::advice::Error::Witness(e)) => e.code(),
            Self::Advice(crate::advice::Error::Opening(e)) => e.code(),
            Self::Advice(crate::advice::Error::Pcs(
                pcs::PcsError::TooLarge { .. } | pcs::PcsError::LayoutMemory,
            )) => ErrorCode::ResMemory,
            Self::Pcs(_) | Self::Advice(_) => ErrorCode::Internal,
        }
    }
}
impl fmt::Display for ProverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())?;
        match self {
            Self::Usage { rule } => write!(f, ": {rule}"),
            Self::Protocol(e) => write!(f, ": {e}"),
            Self::Compile(e) => write!(f, ": {e}"),
            Self::Witness(e) => write!(f, ": {e}"),
            Self::Leaf(e) => write!(f, ": {e}"),
            Self::Stage(e) => write!(f, ": {e}"),
            Self::Pcs(e) => write!(f, ": {e}"),
            Self::Advice(e) => write!(f, ": {e}"),
            Self::Opening(e) => write!(f, ": {e}"),
            Self::Resource(_) => Ok(()),
        }
    }
}
impl std::error::Error for ProverError {}
macro_rules! from_error {
    ($variant:ident, $ty:ty) => {
        impl From<$ty> for ProverError {
            fn from(e: $ty) -> Self {
                Self::$variant(e)
            }
        }
    };
}
from_error!(Compile, tessera_compiler::CompileError);
from_error!(Witness, tessera_witness::WitnessError);
from_error!(Leaf, crate::LeafError);
from_error!(Stage, crate::StageError);
from_error!(Pcs, pcs::PcsError);
from_error!(Advice, crate::advice::Error);
from_error!(Opening, pcs::AdviceOpeningError);
impl From<protocol::SessionError<ProverError>> for ProverError {
    fn from(e: protocol::SessionError<ProverError>) -> Self {
        match e {
            protocol::SessionError::Protocol(e) => Self::Protocol(e),
            protocol::SessionError::Engine(e) => e,
        }
    }
}
