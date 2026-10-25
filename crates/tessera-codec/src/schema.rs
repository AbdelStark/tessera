// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The schema registry: the artifact catalog of 03 §5, with the version each schema's writers
//! emit and the versions its readers accept (09 §3).

use crate::error::SchemaId;

/// Who may see an artifact (03 §5 'Class', 05 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// Anyone.
    Public,
    /// Sent to the verifier, zero-knowledge by construction: it may be shared but not altered.
    Protocol,
    /// The operator's own, such as an execution plan or the verifier's ledger.
    Private,
    /// Secret-class data (05 §3), never in a report.
    Secret,
}

/// A schema of the artifact catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Schema {
    /// The artifact's name, as 03 §5 writes it.
    pub name: &'static str,
    /// The identifier its envelopes carry.
    pub id: SchemaId,
    /// The version writers emit.
    pub version: u16,
    /// The file extension, without its dot; `None` for messages and the digest-log stream.
    pub extension: Option<&'static str>,
    /// Who may see it.
    pub class: Class,
}

impl Schema {
    /// Whether this schema's readers accept `version`: the current one and the one before it
    /// (09 §3). No writer emits version 0, so no reader accepts it.
    pub const fn reads(&self, version: u16) -> bool {
        version != 0
            && (version == self.version
                || matches!(version.checked_add(1), Some(next) if next == self.version))
    }
}

/// Defines one constant per schema and [`CATALOG`] from a single list, so they cannot disagree.
macro_rules! catalog {
    ($($constant:ident = $name:literal, $id:literal, $version:literal, $extension:expr, $class:ident;)*) => {
        $(
            #[doc = concat!("`", $name, "` (03 §5).")]
            pub const $constant: Schema = Schema {
                name: $name,
                id: SchemaId($id),
                version: $version,
                extension: $extension,
                class: Class::$class,
            };
        )*

        /// Every schema, in the order of 03 §5.
        pub const CATALOG: [Schema; [$($id),*].len()] = [$($constant),*];
    };
}

catalog! {
    INTEGER_MODEL = "IntegerModel", 0x0101, 1, Some("d"), Secret;
    CLAIM_LIST = "ClaimList", 0x0201, 1, Some("tcl"), Public;
    EXECUTION_PLAN = "ExecutionPlan", 0x0202, 1, Some("tep"), Private;
    WEIGHT_COMMITMENT_PUBLIC = "WeightCommitmentPublic", 0x0301, 1, Some("twr"), Public;
    WEIGHT_COMMITMENT_STATE = "WeightCommitmentState", 0x0302, 1, Some("twc"), Secret;
    WEIGHT_LINK_PROOF = "WeightLinkProof", 0x0303, 1, Some("twl"), Public;
    DIGEST_LOG_ENTRY = "DigestLogEntry", 0x0401, 1, None, Public;
    LOG_MANIFEST = "LogManifest", 0x0402, 1, Some("tlm"), Public;
    AUDIT_SAMPLE = "AuditSample", 0x0403, 1, Some("tas"), Public;
    SESSION_TRANSCRIPT = "SessionTranscript", 0x0404, 1, Some("tst"), Secret;
    SEED_COMMIT = "SeedCommit", 0x0501, 1, None, Public;
    PROVER_SEED_SHARE = "ProverSeedShare", 0x0502, 1, None, Public;
    SEED_REVEAL = "SeedReveal", 0x0503, 1, None, Public;
    ROUND1 = "Round1", 0x0510, 1, None, Protocol;
    ROUND2_RESPONSE = "Round2Response", 0x0511, 1, None, Protocol;
    COLUMN_QUERIES = "ColumnQueries", 0x0512, 1, None, Protocol;
    ROUND3_RESPONSE = "Round3Response", 0x0513, 1, None, Protocol;
    NI_PROOF = "NiProof", 0x0520, 1, Some("tnp"), Protocol;
    TOKEN_FILE = "TokenFile", 0x0601, 1, Some("ttk"), Secret;
    LOGITS_FILE = "LogitsFile", 0x0602, 1, Some("tlg"), Secret;
    VERIFIER_LEDGER = "VerifierLedger", 0x0701, 1, Some("tvl"), Private;
}

/// The schema with identifier `id`, if the catalog has one.
pub fn by_id(id: SchemaId) -> Option<&'static Schema> {
    CATALOG.iter().find(|schema| schema.id == id)
}
