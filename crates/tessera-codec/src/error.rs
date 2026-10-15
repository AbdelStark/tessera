// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The error taxonomy of docs/spec/04-error-model.md: stable error codes, their classes and
//! exit codes, and [`TesseraError`], whose context can carry public identifiers only.
//!
//! The codes are tier-A identifiers (02 §1). The verifier workspace implements the same codes
//! independently, from the specification and `spec/vectors/errors/codes.json`; nothing here is
//! shared with it (RFC-0001).

use core::fmt;
use std::error::Error;

/// The class of an error code (04 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    /// Invalid arguments or configuration.
    Usage,
    /// A malformed or unsupported input artifact.
    Input,
    /// The model, or its compilation, cannot meet the specification.
    Model,
    /// The honest prover cannot produce a proof for these inputs (04 §1, principle 3).
    Completeness,
    /// A protocol session failed.
    Protocol,
    /// A resource was exhausted or failed.
    Resource,
    /// An invariant was violated: a bug.
    Internal,
}

impl ErrorClass {
    /// The class name as 04 §3 writes it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Usage => "Usage",
            Self::Input => "Input",
            Self::Model => "Model",
            Self::Completeness => "Completeness",
            Self::Protocol => "Protocol",
            Self::Resource => "Resource",
            Self::Internal => "Internal",
        }
    }
}

macro_rules! error_codes {
    ($($variant:ident => $text:literal, $class:ident, $exit:literal, $doc:literal;)*) => {
        /// A stable error code (04 §3). Its [`Display`](fmt::Display) form is the `ERR-...`
        /// identifier.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum ErrorCode {
            $(#[doc = $doc] $variant,)*
        }

        impl ErrorCode {
            /// Every code, in the order of 04 §3.
            pub const ALL: &'static [ErrorCode] = &[$(ErrorCode::$variant),*];

            /// The stable identifier, for example `ERR-CODEC-LIMIT`.
            pub const fn as_str(self) -> &'static str {
                match self { $(ErrorCode::$variant => $text,)* }
            }

            /// The class of the code.
            pub const fn class(self) -> ErrorClass {
                match self { $(ErrorCode::$variant => ErrorClass::$class,)* }
            }

            /// The exit code of the `tessera` command line for this error (02 §2.4, 04 §3).
            pub const fn exit_code(self) -> u8 {
                match self { $(ErrorCode::$variant => $exit,)* }
            }

            /// Parses a stable identifier.
            pub fn from_identifier(text: &str) -> Option<Self> {
                match text { $($text => Some(ErrorCode::$variant),)* _ => None }
            }
        }
    };
}

error_codes! {
    Usage => "ERR-USAGE", Usage, 64, "Invalid command-line arguments or configuration.";
    CodecMagic => "ERR-CODEC-MAGIC", Input, 65, "Envelope magic or schema id mismatch.";
    CodecVersion => "ERR-CODEC-VERSION", Input, 65, "Unsupported schema or protocol version.";
    CodecNoncanonical => "ERR-CODEC-NONCANONICAL", Input, 65,
        "Non-canonical field element, trailing bytes, invalid tag or non-minimal length.";
    CodecTruncated => "ERR-CODEC-TRUNCATED", Input, 65, "Input ended inside a value.";
    CodecDigest => "ERR-CODEC-DIGEST", Input, 65, "Envelope digest does not match the payload.";
    CodecLimit => "ERR-CODEC-LIMIT", Input, 65, "A length exceeds the configured resource limit.";
    QuantArch => "ERR-QUANT-ARCH", Model, 65, "Checkpoint architecture outside `ArchSpec` support.";
    QuantCalib => "ERR-QUANT-CALIB", Model, 20,
        "Calibration produced non-finite or degenerate statistics.";
    CompileRange => "ERR-COMPILE-RANGE", Model, 20,
        "A lookup key cannot fit any gadget's table bound.";
    CompileWrap => "ERR-COMPILE-WRAP", Model, 20, "A lookup key cannot be proven a true integer.";
    CompileProfile => "ERR-COMPILE-PROFILE", Model, 20,
        "The `servable` profile's constraints are unsatisfiable.";
    WitnessExactMismatch => "ERR-WITNESS-EXACT-MISMATCH", Completeness, 20,
        "Exact mode: a recorded output token differs from the output of `D`.";
    WitnessAnchor => "ERR-WITNESS-ANCHOR", Completeness, 20,
        "The prover-held transcript does not match the digest log.";
    WitnessBudget => "ERR-WITNESS-BUDGET", Completeness, 20,
        "The weight-commitment hiding budget is insufficient; a refresh is required.";
    ProtoOrder => "ERR-PROTO-ORDER", Protocol, 20, "Message out of order for the session state.";
    ProtoVersion => "ERR-PROTO-VERSION", Protocol, 20, "The peer speaks another protocol version.";
    ProtoCointoss => "ERR-PROTO-COINTOSS", Protocol, 20,
        "A seed reveal does not match its commitment.";
    ProtoReuse => "ERR-PROTO-REUSE", Protocol, 70,
        "Attempt to answer a second seed for the same `Round1`.";
    ProtoTimeout => "ERR-PROTO-TIMEOUT", Protocol, 20,
        "The peer did not respond within the session deadline.";
    ResMemory => "ERR-RES-MEMORY", Resource, 20, "Memory budget exceeded.";
    ResDevice => "ERR-RES-DEVICE", Resource, 20, "GPU driver or kernel error.";
    ResIo => "ERR-RES-IO", Resource, 20, "Filesystem or network I/O failure.";
    ResTime => "ERR-RES-TIME", Resource, 20, "Wall-time limit exceeded.";
    Internal => "ERR-INTERNAL", Internal, 70, "Invariant violation: a bug.";
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

macro_rules! public_ids {
    ($($name:ident($inner:ty) => $label:literal, $doc:literal;)*) => {$(
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub $inner);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($label, " {}"), self.0)
            }
        }
    )*};
}

public_ids! {
    SchemaId(u16) => "schema", "An artifact schema identifier (03 §5).";
    ByteOffset(u64) => "byte offset", "An offset into an encoded artifact or message.";
    StageId(u16) => "stage", "The index of a batched proof stage (RFC-0010).";
    LayerIndex(u32) => "layer", "A transformer layer, counted from 0.";
    TensorId(u32) => "tensor", "The dense index of a tensor of the compiled graph (03 §3).";
    TableId(u32) => "table", "The dense index of a lookup table (03 §3).";
    GadgetId(u32) => "gadget", "The dense index of a gadget instance (03 §3).";
    RequestId(u64) => "request", "A request identifier from the batch descriptor (03 §4).";
    TokenPosition(u32) => "token position",
        "The position of a token in a request's stream: the position, never the token itself.";
}

/// Where an error happened, in public identifiers only (04 §4, 05 §3).
///
/// Every field is an identifier newtype; none can hold a field element, a token, a tensor, a
/// byte string or free text, so a context cannot carry secret-class data. A new field needs a
/// new identifier type and a look at the redaction rules first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PublicContext {
    /// The artifact schema being decoded.
    pub schema: Option<SchemaId>,
    /// The byte offset where decoding failed.
    pub offset: Option<ByteOffset>,
    /// The proof stage.
    pub stage: Option<StageId>,
    /// The transformer layer.
    pub layer: Option<LayerIndex>,
    /// The tensor.
    pub tensor: Option<TensorId>,
    /// The lookup table.
    pub table: Option<TableId>,
    /// The gadget instance.
    pub gadget: Option<GadgetId>,
    /// The request of the batch.
    pub request: Option<RequestId>,
    /// The position of a token in the request.
    pub position: Option<TokenPosition>,
}

impl PublicContext {
    /// Whether no field is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

impl fmt::Display for PublicContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: [Option<&dyn fmt::Display>; 9] = [
            self.schema.as_ref().map(|v| v as &dyn fmt::Display),
            self.offset.as_ref().map(|v| v as &dyn fmt::Display),
            self.stage.as_ref().map(|v| v as &dyn fmt::Display),
            self.layer.as_ref().map(|v| v as &dyn fmt::Display),
            self.tensor.as_ref().map(|v| v as &dyn fmt::Display),
            self.table.as_ref().map(|v| v as &dyn fmt::Display),
            self.gadget.as_ref().map(|v| v as &dyn fmt::Display),
            self.request.as_ref().map(|v| v as &dyn fmt::Display),
            self.position.as_ref().map(|v| v as &dyn fmt::Display),
        ];
        for (i, part) in parts.iter().flatten().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            part.fmt(f)?;
        }
        Ok(())
    }
}

/// An error that crossed a crate boundary: a stable code, a public context and the error that
/// caused it (04 §4).
///
/// Each crate defines its own error type and converts into this one at its boundary. Every
/// error placed in `source` must itself carry public data only: its message reaches logs
/// and reports like this one's.
#[derive(Debug)]
pub struct TesseraError {
    /// The stable code.
    pub code: ErrorCode,
    /// Where it happened.
    pub context: PublicContext,
    /// The underlying error, if any.
    pub source: Option<Box<dyn Error + Send + Sync>>,
}

impl TesseraError {
    /// An error with `code` and no context.
    pub fn new(code: ErrorCode) -> Self {
        Self {
            code,
            context: PublicContext::default(),
            source: None,
        }
    }

    /// The same error with `context`.
    #[must_use]
    pub fn with_context(mut self, context: PublicContext) -> Self {
        self.context = context;
        self
    }

    /// The same error caused by `source`.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<Box<dyn Error + Send + Sync>>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// The exit code of the `tessera` command line for this error.
    pub fn exit_code(&self) -> u8 {
        self.code.exit_code()
    }
}

impl From<ErrorCode> for TesseraError {
    fn from(code: ErrorCode) -> Self {
        Self::new(code)
    }
}

impl fmt::Display for TesseraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code)?;
        if !self.context.is_empty() {
            write!(f, " ({})", self.context)?;
        }
        Ok(())
    }
}

impl Error for TesseraError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|e| e as &(dyn Error + 'static))
    }
}
