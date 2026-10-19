// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The label registry (RFC-0014 'Transcript schedule'): every tier-A label pattern that
//! `tessera-protocol/1` uses, in the row of the schedule that uses it, and the labels a schedule
//! expands them to.
//!
//! Both sides build every label from these patterns alone (RFC-0004 'Transcript'): a [`Label`]
//! is a pattern with its indices, and its `Display` is the ASCII string the transcript hashes.
//! `{ℓ}`, `{j}`, `{c}` and `{k}` are decimal indices counted from 0, and `{T}` is a weight
//! matrix's name as RFC-0031 writes it (RFC-0036 'Matrix ids').

use core::fmt;

use crate::WeightMatrix;

/// The protocol a schedule is expanded for (RFC-0036 'Versions and envelopes').
///
/// The variant is the typed place where `tessera-protocol/2`'s labels slot in: its masked sums,
/// its LogUp identity sumcheck in place of `logup/roots` and its blinding claims (RFC-0013,
/// RFC-0032; RFC-0014 rows 4 to 7). Until they are defined (#148), a schedule of
/// [`Two`](Self::Two) is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Protocol {
    /// `tessera-protocol/1`, sound and not hiding, whose `Round1.protocol_version` is 0 until it is
    /// assigned (09 §3), then 1.
    One,
    /// `tessera-protocol/2`, which adds the zero-knowledge labels.
    Two,
}

impl Protocol {
    /// The `SessionContext.protocol` string that binds it (RFC-0004 'Transcript').
    pub const fn name(self) -> &'static str {
        match self {
            Self::One => "tessera-protocol/1",
            Self::Two => "tessera-protocol/2",
        }
    }
}

/// Where `tessera-protocol/1` squeezes each weight matrix's proximity point `pcs/weights/{T}/xi`.
///
/// It is the one place the schedule names it: [`WEIGHTS_XI`] chooses, and every schedule follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightsXi {
    /// RFC-0049: right after the seed, in row 3, one label for each weight matrix with committed
    /// rows, in RFC-0040's matrix order. The folding pass then computes each matrix's proximity
    /// row in its one pass over the weights (INV-WC-4); `R_W` is fixed before the session, so a
    /// point drawn after the seed is independent of the committed rows.
    AfterSeed,
    /// RFC-0014 as first written: in row 7, after each opened matrix's `pcs/weights/{T}/gamma`.
    Openings,
}

/// The placement of `pcs/weights/{T}/xi` that every schedule of this crate follows: RFC-0049's,
/// after the seed.
pub const WEIGHTS_XI: WeightsXi = WeightsXi::AfterSeed;

/// A tier-A label pattern of `tessera-protocol/1` (RFC-0014 'Transcript schedule', with the
/// labels RFC-0035, RFC-0037, RFC-0045 and RFC-0049 add or move).
///
/// `tessera-protocol/2`'s patterns (`zk-sum`, `zk-rho`, `logup/identity/…`, the blinding claims)
/// are not here: they come with its schedule (#148).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LabelPattern {
    /// `round1`: `TCE(Round1)`.
    Round1,
    /// `seed`: the coin-tossed seed's 32 bytes in a live session, squeezed in a non-interactive
    /// proof.
    Seed,
    /// `seed-pow`: the seed's proof of work, which absorbs its nonce.
    SeedPow,
    /// `pcs/weights/{T}/xi`: matrix `T`'s proximity point `ξ*_T`, where [`WEIGHTS_XI`] puts it.
    WeightsXi,
    /// `logup/alpha`: `α`.
    LogUpAlpha,
    /// `logup/beta`: `β`.
    LogUpBeta,
    /// `logup/roots`: every LogUp instance's tree roots.
    LogUpRoots,
    /// `tree/{ℓ}/root/{c}`: coordinate `c` of the root point of the forests that join level `ℓ`.
    TreeRoot,
    /// `tree/{ℓ}/gamma`: level `ℓ`'s batching challenge `γ`.
    TreeGamma,
    /// `tree/{ℓ}/lambda`: level `ℓ`'s `λ`, at a level with fraction trees.
    TreeLambda,
    /// `tree/{ℓ}/round/{j}`: level `ℓ`'s round message `j`.
    TreeRound,
    /// `tree/{ℓ}/r/{j}`: level `ℓ`'s round challenge `j`.
    TreeChallenge,
    /// `tree/{ℓ}/evals`: level `ℓ`'s evaluations.
    TreeEvals,
    /// `tree/{ℓ}/line`: level `ℓ`'s line challenge `ρ`.
    TreeLine,
    /// `logup/values`: the lookup trees' value evaluations.
    LogUpValues,
    /// `stage/1/layer/{c}`: coordinate `c` of stage 1's layer point `ρ_1`.
    StageLayer,
    /// `stage/{k}/zero/{c}`: coordinate `c` of stage `k`'s zero-check point.
    StageZero,
    /// `stage/{k}/gamma`: stage `k`'s batching challenge `γ_k`.
    StageGamma,
    /// `stage/{k}/round/{j}`: stage `k`'s round message `j`.
    StageRound,
    /// `stage/{k}/r/{j}`: stage `k`'s round challenge `j`.
    StageChallenge,
    /// `stage/{k}/evals`: stage `k`'s final evaluations.
    StageEvals,
    /// `stage/{k}/split`: stage `k`'s split values; stage 0's are the leaf stage's.
    StageSplit,
    /// `pcs/advice/gamma`: the advice opening's batching challenge `γ`.
    AdviceGamma,
    /// `pcs/advice/xi`: the advice's proximity point `ξ*`.
    AdviceXi,
    /// `pcs/weights/{T}/gamma`: matrix `T`'s `θ_T`.
    WeightsGamma,
    /// `pcs/advice/combined`: the advice's combined rows.
    AdviceCombined,
    /// `pcs/weights/{T}/combined`: matrix `T`'s combined rows.
    WeightsCombined,
    /// `scoring`: the scoring claims.
    Scoring,
    /// `queries`: a live session's `ColumnQueries`.
    Queries,
    /// `query-pow`: the queries' proof of work, which absorbs its nonce.
    QueryPow,
    /// `pcs/advice/queries`: the advice's column indices, squeezed.
    AdviceQueries,
    /// `pcs/weights/{T}/queries`: matrix `T`'s column indices, squeezed.
    WeightsQueries,
}

impl LabelPattern {
    /// Every pattern, in the order of RFC-0014's table, the proofs of work's nonces before what
    /// follows them. `pcs/weights/{T}/xi` is listed in its first place, row 7; [`row`](Self::row)
    /// gives where [`WEIGHTS_XI`] puts it.
    pub const ALL: [Self; 32] = [
        Self::Round1,
        Self::SeedPow,
        Self::Seed,
        Self::LogUpAlpha,
        Self::LogUpBeta,
        Self::LogUpRoots,
        Self::TreeRoot,
        Self::TreeGamma,
        Self::TreeLambda,
        Self::TreeRound,
        Self::TreeChallenge,
        Self::TreeEvals,
        Self::TreeLine,
        Self::LogUpValues,
        Self::StageLayer,
        Self::StageZero,
        Self::StageGamma,
        Self::StageRound,
        Self::StageChallenge,
        Self::StageEvals,
        Self::StageSplit,
        Self::AdviceGamma,
        Self::AdviceXi,
        Self::WeightsGamma,
        Self::WeightsXi,
        Self::AdviceCombined,
        Self::WeightsCombined,
        Self::Scoring,
        Self::Queries,
        Self::QueryPow,
        Self::AdviceQueries,
        Self::WeightsQueries,
    ];

    /// The pattern as RFC-0014 writes it.
    pub const fn template(self) -> &'static str {
        match self {
            Self::Round1 => "round1",
            Self::Seed => "seed",
            Self::SeedPow => "seed-pow",
            Self::WeightsXi => "pcs/weights/{T}/xi",
            Self::LogUpAlpha => "logup/alpha",
            Self::LogUpBeta => "logup/beta",
            Self::LogUpRoots => "logup/roots",
            Self::TreeRoot => "tree/{ℓ}/root/{c}",
            Self::TreeGamma => "tree/{ℓ}/gamma",
            Self::TreeLambda => "tree/{ℓ}/lambda",
            Self::TreeRound => "tree/{ℓ}/round/{j}",
            Self::TreeChallenge => "tree/{ℓ}/r/{j}",
            Self::TreeEvals => "tree/{ℓ}/evals",
            Self::TreeLine => "tree/{ℓ}/line",
            Self::LogUpValues => "logup/values",
            Self::StageLayer => "stage/1/layer/{c}",
            Self::StageZero => "stage/{k}/zero/{c}",
            Self::StageGamma => "stage/{k}/gamma",
            Self::StageRound => "stage/{k}/round/{j}",
            Self::StageChallenge => "stage/{k}/r/{j}",
            Self::StageEvals => "stage/{k}/evals",
            Self::StageSplit => "stage/{k}/split",
            Self::AdviceGamma => "pcs/advice/gamma",
            Self::AdviceXi => "pcs/advice/xi",
            Self::WeightsGamma => "pcs/weights/{T}/gamma",
            Self::AdviceCombined => "pcs/advice/combined",
            Self::WeightsCombined => "pcs/weights/{T}/combined",
            Self::Scoring => "scoring",
            Self::Queries => "queries",
            Self::QueryPow => "query-pow",
            Self::AdviceQueries => "pcs/advice/queries",
            Self::WeightsQueries => "pcs/weights/{T}/queries",
        }
    }

    /// The row of RFC-0014's schedule table that uses it. `pcs/weights/{T}/xi` is in row 3 or 7,
    /// as [`WEIGHTS_XI`] places it, and stage 0's `stage/{k}/split` closes the leaf stage's rows 4
    /// and 5.
    pub const fn row(self) -> u8 {
        match self {
            Self::Round1 => 2,
            Self::Seed | Self::SeedPow => 3,
            Self::WeightsXi => match WEIGHTS_XI {
                WeightsXi::AfterSeed => 3,
                WeightsXi::Openings => 7,
            },
            Self::LogUpAlpha | Self::LogUpBeta | Self::LogUpRoots => 4,
            Self::TreeRoot
            | Self::TreeGamma
            | Self::TreeLambda
            | Self::TreeRound
            | Self::TreeChallenge
            | Self::TreeEvals
            | Self::TreeLine
            | Self::LogUpValues => 5,
            Self::StageLayer
            | Self::StageZero
            | Self::StageGamma
            | Self::StageRound
            | Self::StageChallenge
            | Self::StageEvals
            | Self::StageSplit => 6,
            Self::AdviceGamma | Self::AdviceXi | Self::WeightsGamma => 7,
            Self::AdviceCombined | Self::WeightsCombined => 8,
            Self::Scoring => 9,
            Self::Queries | Self::QueryPow | Self::AdviceQueries | Self::WeightsQueries => 10,
        }
    }
}

/// A label: a pattern of the registry with its indices. Its `Display` is the ASCII string the
/// transcript absorbs or squeezes under.
///
/// Distinct labels display as distinct strings: every pattern has its own fixed parts, and an
/// index is written in decimal or as a matrix's name, neither of which holds a `/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Label {
    /// `round1`.
    Round1,
    /// `seed`.
    Seed,
    /// `seed-pow`.
    SeedPow,
    /// `pcs/weights/{T}/xi`.
    WeightsXi(WeightMatrix),
    /// `logup/alpha`.
    LogUpAlpha,
    /// `logup/beta`.
    LogUpBeta,
    /// `logup/roots`.
    LogUpRoots,
    /// `tree/{ℓ}/root/{c}`.
    TreeRoot {
        /// `ℓ`.
        level: u32,
        /// `c`.
        coord: u32,
    },
    /// `tree/{ℓ}/gamma`.
    TreeGamma {
        /// `ℓ`.
        level: u32,
    },
    /// `tree/{ℓ}/lambda`.
    TreeLambda {
        /// `ℓ`.
        level: u32,
    },
    /// `tree/{ℓ}/round/{j}`.
    TreeRound {
        /// `ℓ`.
        level: u32,
        /// `j`.
        round: u32,
    },
    /// `tree/{ℓ}/r/{j}`.
    TreeChallenge {
        /// `ℓ`.
        level: u32,
        /// `j`.
        round: u32,
    },
    /// `tree/{ℓ}/evals`.
    TreeEvals {
        /// `ℓ`.
        level: u32,
    },
    /// `tree/{ℓ}/line`.
    TreeLine {
        /// `ℓ`.
        level: u32,
    },
    /// `logup/values`.
    LogUpValues,
    /// `stage/1/layer/{c}`.
    StageLayer {
        /// `c`.
        coord: u32,
    },
    /// `stage/{k}/zero/{c}`.
    StageZero {
        /// `k`.
        stage: u16,
        /// `c`.
        coord: u32,
    },
    /// `stage/{k}/gamma`.
    StageGamma {
        /// `k`.
        stage: u16,
    },
    /// `stage/{k}/round/{j}`.
    StageRound {
        /// `k`.
        stage: u16,
        /// `j`.
        round: u32,
    },
    /// `stage/{k}/r/{j}`.
    StageChallenge {
        /// `k`.
        stage: u16,
        /// `j`.
        round: u32,
    },
    /// `stage/{k}/evals`.
    StageEvals {
        /// `k`.
        stage: u16,
    },
    /// `stage/{k}/split`.
    StageSplit {
        /// `k`: 0 for the leaf stage.
        stage: u16,
    },
    /// `pcs/advice/gamma`.
    AdviceGamma,
    /// `pcs/advice/xi`.
    AdviceXi,
    /// `pcs/weights/{T}/gamma`.
    WeightsGamma(WeightMatrix),
    /// `pcs/advice/combined`.
    AdviceCombined,
    /// `pcs/weights/{T}/combined`.
    WeightsCombined(WeightMatrix),
    /// `scoring`.
    Scoring,
    /// `queries`.
    Queries,
    /// `query-pow`.
    QueryPow,
    /// `pcs/advice/queries`.
    AdviceQueries,
    /// `pcs/weights/{T}/queries`.
    WeightsQueries(WeightMatrix),
}

impl Label {
    /// The pattern it expands.
    pub const fn pattern(&self) -> LabelPattern {
        match self {
            Self::Round1 => LabelPattern::Round1,
            Self::Seed => LabelPattern::Seed,
            Self::SeedPow => LabelPattern::SeedPow,
            Self::WeightsXi(_) => LabelPattern::WeightsXi,
            Self::LogUpAlpha => LabelPattern::LogUpAlpha,
            Self::LogUpBeta => LabelPattern::LogUpBeta,
            Self::LogUpRoots => LabelPattern::LogUpRoots,
            Self::TreeRoot { .. } => LabelPattern::TreeRoot,
            Self::TreeGamma { .. } => LabelPattern::TreeGamma,
            Self::TreeLambda { .. } => LabelPattern::TreeLambda,
            Self::TreeRound { .. } => LabelPattern::TreeRound,
            Self::TreeChallenge { .. } => LabelPattern::TreeChallenge,
            Self::TreeEvals { .. } => LabelPattern::TreeEvals,
            Self::TreeLine { .. } => LabelPattern::TreeLine,
            Self::LogUpValues => LabelPattern::LogUpValues,
            Self::StageLayer { .. } => LabelPattern::StageLayer,
            Self::StageZero { .. } => LabelPattern::StageZero,
            Self::StageGamma { .. } => LabelPattern::StageGamma,
            Self::StageRound { .. } => LabelPattern::StageRound,
            Self::StageChallenge { .. } => LabelPattern::StageChallenge,
            Self::StageEvals { .. } => LabelPattern::StageEvals,
            Self::StageSplit { .. } => LabelPattern::StageSplit,
            Self::AdviceGamma => LabelPattern::AdviceGamma,
            Self::AdviceXi => LabelPattern::AdviceXi,
            Self::WeightsGamma(_) => LabelPattern::WeightsGamma,
            Self::AdviceCombined => LabelPattern::AdviceCombined,
            Self::WeightsCombined(_) => LabelPattern::WeightsCombined,
            Self::Scoring => LabelPattern::Scoring,
            Self::Queries => LabelPattern::Queries,
            Self::QueryPow => LabelPattern::QueryPow,
            Self::AdviceQueries => LabelPattern::AdviceQueries,
            Self::WeightsQueries(_) => LabelPattern::WeightsQueries,
        }
    }
}

/// The ASCII label, its pattern with the indices written in.
impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Round1 => f.write_str("round1"),
            Self::Seed => f.write_str("seed"),
            Self::SeedPow => f.write_str("seed-pow"),
            Self::WeightsXi(matrix) => write!(f, "pcs/weights/{matrix}/xi"),
            Self::LogUpAlpha => f.write_str("logup/alpha"),
            Self::LogUpBeta => f.write_str("logup/beta"),
            Self::LogUpRoots => f.write_str("logup/roots"),
            Self::TreeRoot { level, coord } => write!(f, "tree/{level}/root/{coord}"),
            Self::TreeGamma { level } => write!(f, "tree/{level}/gamma"),
            Self::TreeLambda { level } => write!(f, "tree/{level}/lambda"),
            Self::TreeRound { level, round } => write!(f, "tree/{level}/round/{round}"),
            Self::TreeChallenge { level, round } => write!(f, "tree/{level}/r/{round}"),
            Self::TreeEvals { level } => write!(f, "tree/{level}/evals"),
            Self::TreeLine { level } => write!(f, "tree/{level}/line"),
            Self::LogUpValues => f.write_str("logup/values"),
            Self::StageLayer { coord } => write!(f, "stage/1/layer/{coord}"),
            Self::StageZero { stage, coord } => write!(f, "stage/{stage}/zero/{coord}"),
            Self::StageGamma { stage } => write!(f, "stage/{stage}/gamma"),
            Self::StageRound { stage, round } => write!(f, "stage/{stage}/round/{round}"),
            Self::StageChallenge { stage, round } => write!(f, "stage/{stage}/r/{round}"),
            Self::StageEvals { stage } => write!(f, "stage/{stage}/evals"),
            Self::StageSplit { stage } => write!(f, "stage/{stage}/split"),
            Self::AdviceGamma => f.write_str("pcs/advice/gamma"),
            Self::AdviceXi => f.write_str("pcs/advice/xi"),
            Self::WeightsGamma(matrix) => write!(f, "pcs/weights/{matrix}/gamma"),
            Self::AdviceCombined => f.write_str("pcs/advice/combined"),
            Self::WeightsCombined(matrix) => write!(f, "pcs/weights/{matrix}/combined"),
            Self::Scoring => f.write_str("scoring"),
            Self::Queries => f.write_str("queries"),
            Self::QueryPow => f.write_str("query-pow"),
            Self::AdviceQueries => f.write_str("pcs/advice/queries"),
            Self::WeightsQueries(matrix) => write!(f, "pcs/weights/{matrix}/queries"),
        }
    }
}
