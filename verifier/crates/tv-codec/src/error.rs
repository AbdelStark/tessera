// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Decoding errors: a stable code (04 §3) and the offset of the check that failed.

use core::fmt;

/// The 04 §3 codes a decoder reports.
///
/// The six `ERR-CODEC-*` codes are faults of the input, one per failure class of RFC-0003
/// 'Failure modes'. The other codes are faults of the reader, which a verifier reports as
/// `Inconclusive`, never as a rejection of the input (04 §5, RFC-0019 'Check order'):
/// [`is_input_fault`](Self::is_input_fault) tells them apart.
///
/// The identifiers are spelled out in the specification, not shared with the prover's code
/// (04 §4); `spec/vectors/errors/codes.json` pins them for both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// `ERR-CODEC-MAGIC`: wrong envelope magic or schema id.
    Magic,
    /// `ERR-CODEC-VERSION`: unsupported envelope or schema version.
    Version,
    /// `ERR-CODEC-NONCANONICAL`: a non-canonical value, an unknown tag, an unsorted map or bytes
    /// left over.
    Noncanonical,
    /// `ERR-CODEC-TRUNCATED`: the input ends inside a value.
    Truncated,
    /// `ERR-CODEC-DIGEST`: the envelope digest does not match its header and payload.
    Digest,
    /// `ERR-CODEC-LIMIT`: a length, a total size or a nesting depth beyond the decode limits.
    Limit,
    /// `ERR-RES-MEMORY`: memory for the values of a sequence could not be reserved. The reader
    /// ran out of memory; the input is not at fault.
    Memory,
    /// `ERR-RES-TIME`: the reader's configured cooperative deadline expired.
    Time,
    /// `ERR-INTERNAL`: a schema decoder broke its contract, decoding a sequence element from
    /// fewer bytes than the minimum size it declared for it, or from none. A bug in the
    /// verifier, whatever the input.
    Internal,
}

impl ErrorCode {
    /// Every code: the faults of the input in the order of the table in 04 §3, then the faults
    /// of the reader.
    pub const ALL: [ErrorCode; 9] = [
        ErrorCode::Magic,
        ErrorCode::Version,
        ErrorCode::Noncanonical,
        ErrorCode::Truncated,
        ErrorCode::Digest,
        ErrorCode::Limit,
        ErrorCode::Memory,
        ErrorCode::Time,
        ErrorCode::Internal,
    ];

    /// The stable identifier, such as `"ERR-CODEC-TRUNCATED"`.
    pub const fn id(self) -> &'static str {
        match self {
            ErrorCode::Magic => "ERR-CODEC-MAGIC",
            ErrorCode::Version => "ERR-CODEC-VERSION",
            ErrorCode::Noncanonical => "ERR-CODEC-NONCANONICAL",
            ErrorCode::Truncated => "ERR-CODEC-TRUNCATED",
            ErrorCode::Digest => "ERR-CODEC-DIGEST",
            ErrorCode::Limit => "ERR-CODEC-LIMIT",
            ErrorCode::Memory => "ERR-RES-MEMORY",
            ErrorCode::Time => "ERR-RES-TIME",
            ErrorCode::Internal => "ERR-INTERNAL",
        }
    }

    /// Whether the input is at fault, so that a verifier rejects it: true for the
    /// `ERR-CODEC-*` codes, false for memory, time and internal reader faults.
    pub const fn is_input_fault(self) -> bool {
        !matches!(
            self,
            ErrorCode::Memory | ErrorCode::Time | ErrorCode::Internal
        )
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// A decoding failure: which check failed, and where.
///
/// It holds public data only, a code and an offset, never a byte of the input (05 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeError {
    code: ErrorCode,
    offset: u64,
}

impl DecodeError {
    /// The error `code` at `offset` bytes from the start of the input (of the envelope, for an
    /// envelope and its payload).
    pub const fn new(code: ErrorCode, offset: u64) -> Self {
        Self { code, offset }
    }

    /// Which check failed.
    pub const fn code(self) -> ErrorCode {
        self.code
    }

    /// Where it failed, in bytes from the start of the input.
    pub const fn offset(self) -> u64 {
        self.offset
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at offset {}", self.code, self.offset)
    }
}

impl std::error::Error for DecodeError {}
