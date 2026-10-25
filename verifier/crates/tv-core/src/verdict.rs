// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent public verdict/error types (04 sections 2–4, RFC-0043).

use crate::claims::ParamSetId;
use crate::params::ParamSet;
use crate::protocol::FixedBits;
use crate::runtime::Budget;
use crate::validation::{GadgetId, StatementField};
use tv_hash::Digest;
use tv_tables::TableId;

/// Stable operational error codes, independently defined from 04 section 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// `ERR-USAGE` (Usage, exit 64).
    Usage,
    /// `ERR-CODEC-MAGIC` (Input, exit 65).
    CodecMagic,
    /// `ERR-CODEC-VERSION` (Input, exit 65).
    CodecVersion,
    /// `ERR-CODEC-NONCANONICAL` (Input, exit 65).
    CodecNoncanonical,
    /// `ERR-CODEC-TRUNCATED` (Input, exit 65).
    CodecTruncated,
    /// `ERR-CODEC-DIGEST` (Input, exit 65).
    CodecDigest,
    /// `ERR-CODEC-LIMIT` (Input, exit 65).
    CodecLimit,
    /// `ERR-QUANT-ARCH` (Model, exit 65).
    QuantArch,
    /// `ERR-QUANT-CALIB` (Model, exit 20).
    QuantCalib,
    /// `ERR-COMPILE-RANGE` (Model, exit 20).
    CompileRange,
    /// `ERR-COMPILE-WRAP` (Model, exit 20).
    CompileWrap,
    /// `ERR-COMPILE-PROFILE` (Model, exit 20).
    CompileProfile,
    /// `ERR-WITNESS-EXACT-MISMATCH` (Completeness, exit 20).
    WitnessExactMismatch,
    /// `ERR-WITNESS-ANCHOR` (Completeness, exit 20).
    WitnessAnchor,
    /// `ERR-WITNESS-BUDGET` (Completeness, exit 20).
    WitnessBudget,
    /// `ERR-PROTO-ORDER` (Protocol, exit 20).
    ProtocolOrder,
    /// `ERR-PROTO-VERSION` (Protocol, exit 20).
    ProtocolVersion,
    /// `ERR-PROTO-COINTOSS` (Protocol, exit 20).
    ProtocolCoinToss,
    /// `ERR-PROTO-REUSE` (Protocol, exit 70).
    ProtocolReuse,
    /// `ERR-PROTO-TIMEOUT` (Protocol, exit 20).
    ProtocolTimeout,
    /// `ERR-PROTO-QUERIES` (Protocol, exit 20).
    ProtocolQueries,
    /// `ERR-RES-MEMORY` (Resource, exit 20).
    Memory,
    /// `ERR-RES-DEVICE` (Resource, exit 20).
    Device,
    /// `ERR-RES-IO` (Resource, exit 20).
    Io,
    /// `ERR-RES-TIME` (Resource, exit 20).
    Time,
    /// `ERR-INTERNAL` (Internal, exit 70).
    Internal,
}
impl ErrorCode {
    /// Every code, in the specification table order.
    pub const ALL: [Self; 26] = [
        Self::Usage,
        Self::CodecMagic,
        Self::CodecVersion,
        Self::CodecNoncanonical,
        Self::CodecTruncated,
        Self::CodecDigest,
        Self::CodecLimit,
        Self::QuantArch,
        Self::QuantCalib,
        Self::CompileRange,
        Self::CompileWrap,
        Self::CompileProfile,
        Self::WitnessExactMismatch,
        Self::WitnessAnchor,
        Self::WitnessBudget,
        Self::ProtocolOrder,
        Self::ProtocolVersion,
        Self::ProtocolCoinToss,
        Self::ProtocolReuse,
        Self::ProtocolTimeout,
        Self::ProtocolQueries,
        Self::Memory,
        Self::Device,
        Self::Io,
        Self::Time,
        Self::Internal,
    ];
    /// The specification's id.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Usage => "ERR-USAGE",
            Self::CodecMagic => "ERR-CODEC-MAGIC",
            Self::CodecVersion => "ERR-CODEC-VERSION",
            Self::CodecNoncanonical => "ERR-CODEC-NONCANONICAL",
            Self::CodecTruncated => "ERR-CODEC-TRUNCATED",
            Self::CodecDigest => "ERR-CODEC-DIGEST",
            Self::CodecLimit => "ERR-CODEC-LIMIT",
            Self::QuantArch => "ERR-QUANT-ARCH",
            Self::QuantCalib => "ERR-QUANT-CALIB",
            Self::CompileRange => "ERR-COMPILE-RANGE",
            Self::CompileWrap => "ERR-COMPILE-WRAP",
            Self::CompileProfile => "ERR-COMPILE-PROFILE",
            Self::WitnessExactMismatch => "ERR-WITNESS-EXACT-MISMATCH",
            Self::WitnessAnchor => "ERR-WITNESS-ANCHOR",
            Self::WitnessBudget => "ERR-WITNESS-BUDGET",
            Self::ProtocolOrder => "ERR-PROTO-ORDER",
            Self::ProtocolVersion => "ERR-PROTO-VERSION",
            Self::ProtocolCoinToss => "ERR-PROTO-COINTOSS",
            Self::ProtocolReuse => "ERR-PROTO-REUSE",
            Self::ProtocolTimeout => "ERR-PROTO-TIMEOUT",
            Self::ProtocolQueries => "ERR-PROTO-QUERIES",
            Self::Memory => "ERR-RES-MEMORY",
            Self::Device => "ERR-RES-DEVICE",
            Self::Io => "ERR-RES-IO",
            Self::Time => "ERR-RES-TIME",
            Self::Internal => "ERR-INTERNAL",
        }
    }
    /// The specification's class.
    pub const fn class(self) -> &'static str {
        match self {
            Self::Usage => "Usage",
            Self::CodecMagic => "Input",
            Self::CodecVersion => "Input",
            Self::CodecNoncanonical => "Input",
            Self::CodecTruncated => "Input",
            Self::CodecDigest => "Input",
            Self::CodecLimit => "Input",
            Self::QuantArch => "Model",
            Self::QuantCalib => "Model",
            Self::CompileRange => "Model",
            Self::CompileWrap => "Model",
            Self::CompileProfile => "Model",
            Self::WitnessExactMismatch => "Completeness",
            Self::WitnessAnchor => "Completeness",
            Self::WitnessBudget => "Completeness",
            Self::ProtocolOrder => "Protocol",
            Self::ProtocolVersion => "Protocol",
            Self::ProtocolCoinToss => "Protocol",
            Self::ProtocolReuse => "Protocol",
            Self::ProtocolTimeout => "Protocol",
            Self::ProtocolQueries => "Protocol",
            Self::Memory => "Resource",
            Self::Device => "Resource",
            Self::Io => "Resource",
            Self::Time => "Resource",
            Self::Internal => "Internal",
        }
    }
    /// The specification's exit code.
    pub const fn exit_code(self) -> i32 {
        match self {
            Self::Usage => 64,
            Self::CodecMagic => 65,
            Self::CodecVersion => 65,
            Self::CodecNoncanonical => 65,
            Self::CodecTruncated => 65,
            Self::CodecDigest => 65,
            Self::CodecLimit => 65,
            Self::QuantArch => 65,
            Self::QuantCalib => 20,
            Self::CompileRange => 20,
            Self::CompileWrap => 20,
            Self::CompileProfile => 20,
            Self::WitnessExactMismatch => 20,
            Self::WitnessAnchor => 20,
            Self::WitnessBudget => 20,
            Self::ProtocolOrder => 20,
            Self::ProtocolVersion => 20,
            Self::ProtocolCoinToss => 20,
            Self::ProtocolReuse => 70,
            Self::ProtocolTimeout => 20,
            Self::ProtocolQueries => 20,
            Self::Memory => 20,
            Self::Device => 20,
            Self::Io => 20,
            Self::Time => 20,
            Self::Internal => 70,
        }
    }
}
impl From<tv_codec::ErrorCode> for ErrorCode {
    fn from(code: tv_codec::ErrorCode) -> Self {
        use tv_codec::ErrorCode as C;
        match code {
            C::Magic => Self::CodecMagic,
            C::Version => Self::CodecVersion,
            C::Noncanonical => Self::CodecNoncanonical,
            C::Truncated => Self::CodecTruncated,
            C::Digest => Self::CodecDigest,
            C::Limit => Self::CodecLimit,
            C::Memory => Self::Memory,
            C::Internal => Self::Internal,
        }
    }
}
/// The verifier could not evaluate; its code carries no proof values or secrets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VerifierError {
    /// Stable operational code, distinct from a proof rejection.
    pub code: ErrorCode,
}
impl core::fmt::Display for VerifierError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.code.id())
    }
}
impl std::error::Error for VerifierError {}
/// The outer artifact's schema id, including when an inner message fails (RFC-0043).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactKind(pub u16);
/// The PCS commitment class of 03 section 3, not a prover-selected matrix id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommitmentClass {
    /// Per-proof advice.
    Advice,
    /// Persistent weights, including ChannelParams.
    Weight,
}
/// The failed non-interactive proof of work (RFC-0043).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PowPhase {
    /// Before the seed.
    Seed,
    /// After all combined rows and scoring claims.
    Queries,
}
/// The anchor check that failed (RFC-0043 'Anchor', with RFC-0059, RFC-0061, RFC-0062 and
/// RFC-0063). Its TCE encoding is a `u16` tag ([`tag`](Self::tag)): tag 1, `Inclusion`, is retired
/// by RFC-0062 and stays reserved, so no variant has it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnchorFailure {
    /// Tag 0, RFC-0015 check 1: the manifest's signature, its scheme or its key, or the
    /// manifest's other checks alone (RFC-0060 'The chain'). It names no request.
    ManifestSignature,
    /// Tag 2, check 2: a request's frames, not contiguous from 0, not its lengths, or of an
    /// impossible length (RFC-0059); two frames of the session in the period at one position or
    /// with one counter (RFC-0062); no end or two, an end in the previous period, a frame after it
    /// or none before it, a late inbound frame, or a position logged twice across the boundary
    /// (RFC-0063).
    Frames,
    /// Tag 3, check 2: a request of a temperature batch that falls within no pin, or within
    /// another than the batch's first request's (RFC-0061).
    SamplerKey,
    /// Tag 4, check 2: the request's session, in a period holding a frame of the request, has no
    /// open or carry, several, or one after another of its entries; or two key commitments across
    /// the request's two periods (RFC-0062, RFC-0063).
    Session,
}
impl AnchorFailure {
    /// The TCE tag: the variant's index in RFC-0043's declaration, which keeps the retired tag 1.
    pub const fn tag(self) -> u16 {
        match self {
            Self::ManifestSignature => 0,
            Self::Frames => 2,
            Self::SamplerKey => 3,
            Self::Session => 4,
        }
    }
}
/// The first failed proof predicate in RFC-0019 order; all payloads are public identifiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// `REJ-MALFORMED`: the prover's outer artifact failed syntax or fixed shape.
    Malformed {
        /// Outer prover artifact schema.
        artifact: ArtifactKind,

        /// Stable input failure code.
        code: ErrorCode,
    },
    /// `REJ-STATEMENT`: the first mismatching public field.
    StatementMismatch {
        /// First public field that fails.
        field: StatementField,
    },
    /// `REJ-SUMCHECK`: reduction stage and exact rejected round.
    Sumcheck {
        /// Reduction stage index.
        stage: u16,
        /// Exact round, or the final-check index.
        round: u32,
    },
    /// `REJ-TREE`: a leaf stage's tree level and exact rejected round (RFC-0033, RFC-0054
    /// 'Messages').
    TreeLevel {
        /// Whose levels: `h` for group `h`'s, `G` for a grouped stage's table batch, 0 for an
        /// ungrouped stage.
        group: u32,
        /// Level from the root of its batch.
        level: u32,
        /// Exact round, or the final-check index.
        round: u32,
    },
    /// `REJ-LEAF-MERGE`: a grouped leaf stage's merge (RFC-0054 'The merge').
    LeafMerge {
        /// Exact round, or `n_max` for the final check.
        round: u32,
    },
    /// `REJ-LOGUP-IDENTITY`: protocol/2 only.
    LogUpIdentity {
        /// Exact round, or the final-check index.
        round: u32,
    },
    /// `REJ-MASK-BATCH`: protocol/2 only; the batching of every mask claim, `masks/round/{j}`
    /// (RFC-0070 'The batching sumcheck', 'Rejections').
    MaskBatch {
        /// Exact round from 0, as the labels count, or `m`, the number of variables, for the
        /// end: a round past the last, a missing `y` or a false final check.
        round: u32,
    },
    /// `REJ-LOOKUP`: verifier-derived table identity.
    Lookup {
        /// Verifier-derived table id.
        table: TableId,
    },
    /// `REJ-GADGET`: verifier-derived gadget identity.
    Gadget {
        /// Verifier-derived gadget id.
        gadget: GadgetId,
    },
    /// `REJ-PCS-MERKLE`: first queried column whose authentication fails.
    MerklePath {
        /// Advice or persistent weights.
        commitment: CommitmentClass,

        /// Smallest failing queried column.
        column: u32,
    },
    /// `REJ-PCS-PROXIMITY`: first inconsistent queried column.
    Proximity {
        /// Advice or persistent weights.
        commitment: CommitmentClass,

        /// Smallest failing queried column.
        column: u32,
    },
    /// `REJ-PCS-EVAL`: original opening obligations fail.
    Evaluation {
        /// Advice or persistent weights.
        commitment: CommitmentClass,
    },
    /// `REJ-POW`: invalid seed or query nonce.
    ProofOfWork {
        /// Seed or query proof of work.
        phase: PowPhase,
    },
    /// `REJ-ANCHOR`: the first anchor check that failed (RFC-0043 'Anchor').
    Anchor {
        /// The request being checked, or None for the manifest's signature.
        request_id: Option<u64>,

        /// First failed anchor predicate.
        kind: AnchorFailure,
    },
    /// `REJ-UHAT`: per-request or batch bound exceeded.
    UBoundExceeded {
        /// Request id, or None for a batch-level failure.
        request_id: Option<u64>,

        /// Claimed fixed-point unexplained bits.
        uhat: FixedBits,

        /// Applicable fixed-point upper bound.
        bound: FixedBits,
    },
    /// `REJ-BUDGET`: persistent epoch/blinding reservation failure, not a preset target.
    Budget {
        /// Trusted commitment epoch.
        epoch: u32,
    },
    /// `REJ-PARAMS`: unsupported name, mode, layout or failed runtime target.
    Params {
        /// Exact versioned parameter-set name.
        name: ParamSetId,
    },
}
impl RejectReason {
    /// Stable rejection identifier (04 section 2).
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Malformed { .. } => "REJ-MALFORMED",
            Self::StatementMismatch { .. } => "REJ-STATEMENT",
            Self::Sumcheck { .. } => "REJ-SUMCHECK",
            Self::TreeLevel { .. } => "REJ-TREE",
            Self::LeafMerge { .. } => "REJ-LEAF-MERGE",
            Self::LogUpIdentity { .. } => "REJ-LOGUP-IDENTITY",
            Self::MaskBatch { .. } => "REJ-MASK-BATCH",
            Self::Lookup { .. } => "REJ-LOOKUP",
            Self::Gadget { .. } => "REJ-GADGET",
            Self::MerklePath { .. } => "REJ-PCS-MERKLE",
            Self::Proximity { .. } => "REJ-PCS-PROXIMITY",
            Self::Evaluation { .. } => "REJ-PCS-EVAL",
            Self::ProofOfWork { .. } => "REJ-POW",
            Self::Anchor { .. } => "REJ-ANCHOR",
            Self::UBoundExceeded { .. } => "REJ-UHAT",
            Self::Budget { .. } => "REJ-BUDGET",
            Self::Params { .. } => "REJ-PARAMS",
        }
    }
}
/// Completed protocol/1 verification. Test parameters certify nothing; no ZK/anchor claim.
#[derive(Debug)]
pub struct AcceptReport {
    /// Accepted versioned preset.
    pub params: ParamSet,
    /// Digest of the trusted public statement.
    pub statement_digest: Digest,
    /// Digest of the validated claim list.
    pub claim_list_digest: Digest,
    /// Bound trusted weight root.
    pub weights_root: Digest,
    /// Bound manifest digest; anchor signatures are outside v0.3.
    pub manifest_digest: Digest,
    /// Exact per-request unexplained-information bound in batch order.
    pub requests: Vec<(u64, FixedBits)>,
    /// Sum of the exact per-request bounds.
    pub batch_uhat: FixedBits,
    /// The complete mandatory runtime soundness calculation.
    pub budget: Budget,
}
/// Only Accept certifies the supported statement; rejection is distinct from inability to judge.
#[derive(Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "return the complete report by value without an infallible box allocation"
)]
pub enum Verdict {
    /// Every supported proof/session predicate passed.
    Accept(AcceptReport),
    /// The proof failed its first predicate.
    Reject(RejectReason),
    /// Own input, resource or operational failure; no proof was judged.
    Inconclusive(VerifierError),
}
impl Verdict {
    /// CLI exit code, retaining the operational taxonomy (02 section 2.4).
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Accept(_) => 0,
            Self::Reject(_) => 10,
            Self::Inconclusive(error) => error.code.exit_code(),
        }
    }
}

// Fallible formatting keeps report allocation failure out of the process-abort path.
pub(crate) struct Json(pub(crate) String);
impl core::fmt::Write for Json {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        self.0
            .try_reserve(text.len())
            .map_err(|_| core::fmt::Error)?;
        self.0.push_str(text);
        Ok(())
    }
}
impl Json {
    pub(crate) fn quoted(&mut self, text: &str) -> core::fmt::Result {
        use core::fmt::Write;
        self.write_char('"')?;
        for ch in text.chars() {
            match ch {
                '"' => self.write_str("\\\"")?,
                '\\' => self.write_str("\\\\")?,
                c if c < ' ' => write!(self, "\\u{:04x}", u32::from(c))?,
                c => self.write_char(c)?,
            }
        }
        self.write_char('"')
    }
    pub(crate) fn digest(&mut self, digest: &Digest) -> core::fmt::Result {
        use core::fmt::Write;
        self.write_char('"')?;
        for b in digest {
            write!(self, "{b:02x}")?;
        }
        self.write_char('"')
    }
}
impl RejectReason {
    pub(crate) fn payload(&self, out: &mut Json) -> core::fmt::Result {
        use core::fmt::Write;
        match self {
            Self::Malformed { artifact, code } => write!(
                out,
                "\"artifact\":{},\"error\":\"{}\"",
                artifact.0,
                code.id()
            ),
            Self::StatementMismatch { field } => write!(out, "\"field\":\"{field}\""),
            Self::Sumcheck { stage, round } => write!(out, "\"stage\":{stage},\"round\":{round}"),
            Self::TreeLevel {
                group,
                level,
                round,
            } => write!(out, "\"group\":{group},\"level\":{level},\"round\":{round}"),
            Self::LeafMerge { round }
            | Self::LogUpIdentity { round }
            | Self::MaskBatch { round } => {
                write!(out, "\"round\":{round}")
            }
            Self::Lookup { table } => write!(out, "\"table\":{}", table.0),
            Self::Gadget { gadget } => write!(out, "\"gadget\":{}", gadget.0),
            Self::MerklePath { commitment, column } | Self::Proximity { commitment, column } => {
                write!(out, "\"commitment\":\"{commitment:?}\",\"column\":{column}")
            }
            Self::Evaluation { commitment } => write!(out, "\"commitment\":\"{commitment:?}\""),
            Self::ProofOfWork { phase } => write!(out, "\"phase\":\"{phase:?}\""),
            Self::Anchor { request_id, kind } => {
                out.write_str("\"request_id\":")?;
                if let Some(id) = request_id {
                    write!(out, "{id}")?;
                } else {
                    out.write_str("null")?;
                }
                write!(out, ",\"kind\":\"{kind:?}\"")
            }
            Self::UBoundExceeded {
                request_id,
                uhat,
                bound,
            } => {
                out.write_str("\"request_id\":")?;
                if let Some(id) = request_id {
                    write!(out, "{id}")?;
                } else {
                    out.write_str("null")?;
                }
                write!(
                    out,
                    ",\"uhat_fixed\":{},\"bound_fixed\":{}",
                    uhat.0, bound.0
                )
            }
            Self::Budget { epoch } => write!(out, "\"epoch\":{epoch}"),
            Self::Params { name } => {
                out.write_str("\"name\":")?;
                out.quoted(name.as_str())
            }
        }
    }
}
impl Verdict {
    /// Deterministic public core report, with no protocol evaluations or measured timings.
    /// Accepted v0.3 reports explicitly exclude anchor, ledger and zero-knowledge claims.
    /// A rejection/error has no acceptance metadata; the CLI can attach its own audit trace.
    ///
    /// # Errors
    /// `ERR-RES-MEMORY` when report text cannot be reserved.
    pub fn to_json(&self) -> Result<String, VerifierError> {
        let mut out = Json(String::new());
        self.write_json(&mut out).map_err(|_| VerifierError {
            code: ErrorCode::Memory,
        })?;
        Ok(out.0)
    }
    fn write_json(&self, out: &mut Json) -> core::fmt::Result {
        use core::fmt::Write;
        out.write_str("{\"schema\":\"tessera.verdict/1\",\"protocol\":\"tessera-protocol/1\",")?;
        match self {
            Self::Accept(report) => {
                out.write_str("\"params\":")?;
                out.quoted(report.params.name())?;
                write!(
                    out,
                    ",\"test_parameters\":{},\"anchor_verified\":false,\"ledger_updated\":false,\"zero_knowledge\":false,\"digests\":{{\"statement\":",
                    report.params.name() == "test-small/1"
                )?;
                out.digest(&report.statement_digest)?;
                out.write_str(",\"claim_list\":")?;
                out.digest(&report.claim_list_digest)?;
                out.write_str(",\"weights\":")?;
                out.digest(&report.weights_root)?;
                out.write_str(",\"manifest\":")?;
                out.digest(&report.manifest_digest)?;
                out.write_str("},\"checks\":[")?;
                for (i, name) in [
                    "input",
                    "params",
                    "claims",
                    "statement",
                    "weight_reservation",
                    "transcript",
                    "leaf",
                    "stages",
                    "evaluations",
                    "queries",
                    "columns",
                    "exact_bound",
                    "complete",
                ]
                .into_iter()
                .enumerate()
                {
                    if i != 0 {
                        out.write_char(',')?;
                    }
                    let outcome = if i == 4 { "not_applicable" } else { "pass" };
                    write!(
                        out,
                        "{{\"step\":{},\"check\":\"{name}\",\"outcome\":\"{outcome}\",\"duration_ns\":null}}",
                        i.saturating_add(1)
                    )?;
                }
                out.write_str("],\"requests\":[")?;
                for (i, (id, u)) in report.requests.iter().enumerate() {
                    if i != 0 {
                        out.write_char(',')?;
                    }
                    write!(out, "{{\"request_id\":{id},\"uhat_fixed\":{}}}", u.0)?;
                }
                write!(
                    out,
                    "],\"batch_uhat_fixed\":{},\"verdict\":{{\"status\":\"accept\"}}",
                    report.batch_uhat.0
                )?;
            }
            Self::Reject(reason) => {
                out.write_str("\"params\":null,\"digests\":null,\"checks\":[],\"requests\":null,\"batch_uhat_fixed\":null,\"verdict\":{\"status\":\"reject\",\"reason\":{")?;
                write!(out, "\"code\":\"{}\",", reason.code())?;
                reason.payload(out)?;
                out.write_str("}}")?;
            }
            Self::Inconclusive(error) => {
                write!(
                    out,
                    "\"params\":null,\"digests\":null,\"checks\":[],\"requests\":null,\"batch_uhat_fixed\":null,\"verdict\":{{\"status\":\"inconclusive\",\"error\":{{\"code\":\"{}\"}}}}",
                    error.code.id()
                )?;
            }
        }
        out.write_str("}\n")
    }
}
