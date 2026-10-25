// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The soundness calculator (RFC-0014 'Soundness budget', 'Soundness calculator'): every row of
//! the soundness table for a parameter set at a statement's sizes, with its error bound, its bits
//! as a FixedBits lower bound, its target, and whether it passes, exactly.
//!
//! A row's bits are `λ = g + max(0, −log2 ε)` for its error bound `ε` and the `g` bits of
//! grinding that cover it. The calculator computes with integers only ([`exact`]): its FixedBits
//! `r` satisfies `r ≤ 256·λ < r + 1 + 2^{-20}`, and a row passes exactly when `λ ≥ t` for its
//! target `t`, decided by comparing integers, never rounded bits.
//!
//! | Row | Error bound `ε` | Grinding | Target |
//! |---|---|---|---|
//! | [`Rounds`](RowKind::Rounds) | `D/q`, `D` the largest challenge error of the stages, the tree levels and, with token groups, the merge | — | `bits` |
//! | [`Union`](RowKind::Union) | `U/q`, `U` the sum of them all, with token groups `G` more copies of the leaf stage's levels and the merge ([`LeafGroupSizes`]) | — | `bits` |
//! | [`Forests`](RowKind::Forests) | `s/q` for the largest forest of `2^s` trees | — | `bits` |
//! | [`LogUp`](RowKind::LogUp) | `(N + T)/q`, `2(N + T)/q` in `tessera-protocol/2`, the largest instance | `g_seed` | `one_shot` |
//! | [`Openings`](RowKind::Openings) | `T_g/q` for the largest claim group, `1/q` per weight matrix | — | `bits` |
//! | [`Queries`](RowKind::Queries) | `((n + k + P)/(2n))^τ` per commitment | `g_query` | `one_shot` |
//! | [`Field`](RowKind::Field) | `ϑ·n/q` per commitment | — | `bits` |
//! | [`Linking`](RowKind::Linking) | `(ν + 1)/q` for the matrix of most variables: its evaluation, and the link's combining challenge `θ_T` | — | `bits` |
//! | [`Hash`](RowKind::Hash) | `2^{-128}`, BLAKE3's 256-bit output | — | `bits` |
//!
//! Here `q = |Fp2| = p^2`, and the targets are the parameter set's ([`Targets`]).
//!
//! A weight matrix's `P` and `ϑ` depend on the protocol (RFC-0049 'The calculator'):
//! `tessera-protocol/1` commits no blinding row and no mask for the weights, so `P = 0` and
//! `ϑ_T = ⌈log2 m_T⌉`, and a matrix with no committed row has neither row; `tessera-protocol/2`
//! counts `P_W` and `ϑ_T = ⌈log2(m_T + B_W)⌉`.

mod exact;
mod llama;

use crate::{FixedBits, ParamSet, ProtocolError, Targets, WeightMatrix};

use exact::UNIT;

/// The most column queries a codeword takes, as [`QueryShape`](crate::QueryShape) allows.
const MAX_QUERIES: u32 = 1 << 16;

/// The longest codeword a `u32` index reaches.
const MAX_CODEWORD: u64 = 1 << 32;

/// Rounds of one degree, of any stage or tree level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rounds {
    /// How many rounds.
    pub count: u64,
    /// Their round polynomials' degree `d`: each round errs with probability `d/q`.
    pub degree: u32,
}

/// One LogUp instance (RFC-0008 'Relation').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LookupSizes {
    /// `N`, its lookups.
    pub lookups: u64,
    /// `T`, its table's rows.
    pub rows: u64,
}

/// One opened weight matrix (RFC-0012 'Layout', RFC-0040 'The matrices').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixSizes {
    /// The matrix.
    pub matrix: WeightMatrix,
    /// `k_T`, its message width: its codeword holds `n_T = k_T·2^{log2_inv_rate}` symbols.
    pub width: u64,
    /// `m_T`, its committed data rows. In `tessera-protocol/1`, which commits no blinding row and
    /// no mask for the weights, they give `ϑ_T = ⌈log2 m_T⌉`, and a matrix with none has no
    /// proximity rows (RFC-0049 'The calculator'); in `tessera-protocol/2`, with its
    /// [`blinding_rows`](Self::blinding_rows) `B_T`, `ϑ_T = ⌈log2(m_T + B_T)⌉`.
    pub rows: u64,
    /// `ν_T`, the variables of its multilinear polynomial, which the linking proof opens.
    pub variables: u32,
    /// `e_T`, its claims in the pool (RFC-0065 'The count'): 1 for `Embed`..`LmHead`, `e_CP` for
    /// `ChannelParams`. `tessera-protocol/1` reads it not.
    pub claims: u32,
}

impl MatrixSizes {
    /// `B_T`, its blinding rows in `tessera-protocol/2` under `B_W`, the parameter set's (RFC-0065
    /// '`ChannelParams`' blinding rows'): `B_W` for `Embed`..`LmHead`, and for `ChannelParams`
    /// `B_W + 2·(e_CP − 1)·⌊max(B_W − 4, 0)/6⌋`, 0 when `e_CP = 0`. Exact in `u128`: below
    /// `2^{64}`.
    pub fn blinding_rows(&self, b_w: u32) -> u128 {
        let b_w = u128::from(b_w);
        match (self.matrix, self.claims) {
            (WeightMatrix::ChannelParams, 0) => 0,
            (WeightMatrix::ChannelParams, e) => {
                b_w + 2 * u128::from(e - 1) * (b_w.saturating_sub(4) / 6)
            }
            _ => b_w,
        }
    }
}

/// The sizes of a statement that the soundness bounds count (RFC-0014 'Soundness calculator').
///
/// A schedule's stages and tree levels are batched sumchecks. Each stage of `n` rounds of degree
/// `d` batching `m` instances errs with probability at most `(n·d + m)/q`, a tree level at most
/// `(n·d + m + 1)/q` with its line challenge, each `ZeroCheck` of `n_i` variables adds `n_i/q`,
/// and in `tessera-protocol/2` each masked sumcheck adds `1/q` for its `ρ` (RFC-0010 'Soundness
/// of the schedule', RFC-0013, RFC-0035).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StatementSizes {
    /// The `n` of `tessera-protocol/n`, whose bounds apply: 1, or 2 with the zero-knowledge items.
    pub protocol: u16,
    /// Every round of every stage and tree level, by degree; in `tessera-protocol/2`, the LogUp
    /// identity sumcheck's too.
    pub rounds: Vec<Rounds>,
    /// The instances each reduction stage batches, one entry per stage; in
    /// `tessera-protocol/2`, the LogUp identity sumcheck is one more.
    pub stages: Vec<u64>,
    /// The instances each tree level batches: the fraction trees and product forests at it.
    pub tree_levels: Vec<u64>,
    /// The variables `n_i` of each `ZeroCheck` (RFC-0035 'Zero-check points').
    pub zero_checks: Vec<u32>,
    /// Each product forest's `s`: it holds `2^s` trees (RFC-0010 'Tree levels').
    pub forests: Vec<u32>,
    /// Each LogUp instance.
    pub lookups: Vec<LookupSizes>,
    /// The claims `T_g` of each advice claim group (RFC-0011 'Opening protocol').
    pub claim_groups: Vec<u64>,
    /// The rows the advice matrix's row index spans, `2^{ν − log2 k}` (RFC-0011 'Canonical
    /// advice layout'): `ϑ_A = ⌈log2 advice_rows⌉`.
    pub advice_rows: u64,
    /// The opened weight matrices, in increasing [`WeightMatrix`] order.
    pub matrices: Vec<MatrixSizes>,
    /// The leaf stage in token groups, at the most groups the verifier accepts for the statement
    /// (RFC-0054 'Soundness').
    pub groups: LeafGroupSizes,
}

/// The leaf stage in token groups (RFC-0054 'Soundness'). A cheating prover chooses its grouping,
/// so the bounds count the most groups the verifier accepts for the statement,
/// `G = 2^{min(14, t + n_ℓ)}`, and none in `tessera-protocol/2`.
///
/// Each group's levels and the table batch's cost at most the ungrouped leaf stage's: the union
/// row counts `G` more copies of them, `G·(Σ level_rounds count·d + Σ_levels (m + 1))`, and the
/// merge's `2·n_max + P + G`. The rounds row takes the merge's degree 2 and its batching
/// `P + G`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LeafGroupSizes {
    /// `G`: a power of two up to `2^14`, and 1 when the verifier accepts no grouping.
    pub count: u64,
    /// The ungrouped leaf stage's level rounds, by degree, which [`StatementSizes::rounds`] also
    /// counts once; empty when `G = 1`.
    pub level_rounds: Vec<Rounds>,
    /// `n_max`, the merge's rounds: the most variables of a merged polynomial; 0 when `G = 1`.
    pub merge_rounds: u32,
    /// `P`, the merged polynomials' count, one per ungrouped split value and per forest, which
    /// bounds every group's claims; 0 when `G = 1`.
    pub merge_claims: u64,
}

impl LeafGroupSizes {
    /// No grouping: the verifier accepts the leaf stage in one batch only.
    pub const NONE: Self = Self {
        count: 1,
        level_rounds: Vec::new(),
        merge_rounds: 0,
        merge_claims: 0,
    };

    /// The most groups RFC-0054 'Messages' admits, `2^14`.
    pub const MAX_COUNT: u64 = 1 << 14;

    /// Whether the verifier accepts a grouping: `G > 1`.
    pub fn grouped(&self) -> bool {
        self.count > 1
    }
}

/// The advice commitment or one weight matrix's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Commitment {
    /// The streaming advice commitment (RFC-0011).
    Advice,
    /// A weight matrix's commitment (RFC-0012).
    Weights(WeightMatrix),
}

/// A row of RFC-0014's soundness table. Tuple compression has no row of its own: the LogUp
/// bound includes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowKind {
    /// Sumcheck and tree rounds: the largest error of a single challenge the stages and tree
    /// levels squeeze, their rounds', batching challenges', zero-check points', line challenges'
    /// and, in `tessera-protocol/2`, masks'.
    Rounds,
    /// All rounds, union bound: the sum of those errors.
    Union,
    /// Product-forest root claims.
    Forests,
    /// The LogUp identity.
    LogUp,
    /// Opening evaluation checks.
    Openings,
    /// Proximity: a commitment's column queries.
    Queries(Commitment),
    /// Proximity: a commitment's field term, the tensor generator's.
    Field(Commitment),
    /// Weight linking.
    Linking,
    /// Hash collisions.
    Hash,
}

/// A proof of work that covers a row (RFC-0014 'Grinding').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProofOfWork {
    /// `seed-pow`, before the seed, from which `α` and `β` follow at once.
    Seed,
    /// `query-pow`, before the column queries.
    Query,
}

/// A row's error bound, exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorBound {
    /// `numerator/q`, `q = |Fp2|`. Numerators saturate at `2^128 − 1`, past `q`.
    Field {
        /// The numerator.
        numerator: u128,
    },
    /// `((n + k + P)/(2n))^τ = ((1 + ρ')/2)^τ`, `ρ' = (k + P)/n`.
    Queries {
        /// `n`, the codeword's symbols.
        codeword: u64,
        /// `k`, the message's.
        message: u64,
        /// `P`, each row's mask coefficients: 0 for a weight matrix in `tessera-protocol/1`.
        mask: u64,
        /// `τ`, the queries.
        queries: u32,
    },
    /// `2^{-128}`.
    Hash,
}

/// A row of a soundness report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SoundnessRow {
    /// Which row.
    pub kind: RowKind,
    /// Its error bound.
    pub bound: ErrorBound,
    /// A field term's `ϑ`, the commitment's row-index bits.
    pub theta: Option<u32>,
    /// A queries row's per-query bits, `−log2((1 + ρ')/2)`, or 0 when `ρ' ≥ 1`.
    pub per_query: Option<FixedBits>,
    /// `λ = g + max(0, −log2 ε)` as a FixedBits lower bound, or `None` when the row has no term:
    /// its error is 0.
    pub bits: Option<FixedBits>,
    /// `g`, the bits of the proof of work that covers it: 0 for the rows none covers.
    pub grinding: u32,
    /// The proof of work that covers it.
    pub covered_by: Option<ProofOfWork>,
    /// The bits it must reach, or `None` under a set held to no target.
    pub target: Option<u32>,
    /// Whether `λ ≥ target`, exactly; a row without a term passes.
    pub pass: Option<bool>,
}

impl SoundnessRow {
    /// The bits above its target, in FixedBits units: negative when it fails, or `None` without
    /// bits or target. A row may pass with a margin of `−1` when `λ` lies within `2^{-28}` above
    /// its target, where the lower bound rounds below it.
    pub fn margin(&self) -> Option<i64> {
        let bits = i64::try_from(self.bits?.0).ok()?;
        Some(bits - i64::from(self.target?) * UNIT as i64)
    }
}

/// The soundness report of a parameter set at a statement's sizes (RFC-0014 'Soundness
/// calculator').
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SoundnessReport {
    /// The parameter set.
    pub params: ParamSet,
    /// The statement's sizes.
    pub sizes: StatementSizes,
    /// The rows, in RFC-0014's order: rounds, union, forests, LogUp, openings, then the advice's
    /// queries and field term and each matrix's in order, linking, and hash.
    pub rows: Vec<SoundnessRow>,
}

impl SoundnessReport {
    /// Every row of `params` at `sizes`.
    ///
    /// # Errors
    ///
    /// [`ProtocolError::Sizes`] for sizes or parameters it cannot bound: a protocol other than
    /// 1 or 2; `τ_A` or `τ_W` outside `1..=2^16`; a codeword of more than `2^32` symbols or none;
    /// no advice row, or in `tessera-protocol/2` a matrix with no data or blinding row, no claim,
    /// or, other than `ChannelParams`, claims other than 1; matrices out of [`WeightMatrix`]
    /// order.
    /// In `tessera-protocol/1` a matrix with no committed row is bounded: it has no queries row
    /// and no field term (RFC-0049).
    pub fn new(params: &ParamSet, sizes: &StatementSizes) -> Result<Self, ProtocolError> {
        let geometry = validate(params, sizes)?;
        let builder = Builder {
            targets: params.targets,
        };
        let v2 = sizes.protocol == 2;
        let grouped = sizes.groups.grouped();
        let batched = sizes.stages.len() as u128 + sizes.tree_levels.len() as u128;
        let mut rows = Vec::with_capacity(7 + 2 * sizes.matrices.len());

        // The largest error of one challenge of the stages and tree levels.
        let largest = sizes
            .rounds
            .iter()
            .filter(|r| r.count > 0)
            .map(|r| u128::from(r.degree))
            .chain(
                sizes
                    .stages
                    .iter()
                    .chain(&sizes.tree_levels)
                    .map(|&m| u128::from(m)),
            )
            .chain(sizes.zero_checks.iter().map(|&z| u128::from(z)))
            // A tree level's line challenge, and in tessera-protocol/2 a mask's ρ: 1/q each.
            .chain((!sizes.tree_levels.is_empty()).then_some(1))
            .chain((v2 && batched > 0).then_some(1))
            // RFC-0054 'Soundness': the groups' levels, the merge's degree 2, and its batching
            // (P + G)/q.
            .chain(
                grouped
                    .then(|| group_challenges(&sizes.groups))
                    .into_iter()
                    .flatten(),
            )
            .max()
            .unwrap_or(0);
        rows.push(builder.field(RowKind::Rounds, largest, None, None));

        // Their sum: (n·d + m) per stage, (n·d + m + 1) per tree level.
        let union = [
            sum(sizes
                .rounds
                .iter()
                .map(|r| u128::from(r.count) * u128::from(r.degree))),
            sum(sizes.stages.iter().map(|&m| u128::from(m))),
            sum(sizes.tree_levels.iter().map(|&m| u128::from(m) + 1)),
            sum(sizes.zero_checks.iter().map(|&z| u128::from(z))),
            if v2 { batched } else { 0 },
            if grouped { group_union(sizes) } else { 0 },
        ]
        .into_iter()
        .fold(0, u128::saturating_add);
        rows.push(builder.field(RowKind::Union, union, None, None));

        let forests = sizes.forests.iter().map(|&s| u128::from(s)).max();
        rows.push(builder.field(RowKind::Forests, forests.unwrap_or(0), None, None));

        // The largest instance: per instance, a false lookup passes with probability (N + T)/q,
        // and twice that in tessera-protocol/2, which also counts a zero denominator (RFC-0032).
        let instances = sizes
            .lookups
            .iter()
            .map(|l| u128::from(l.lookups) + u128::from(l.rows))
            .max()
            .unwrap_or(0);
        let logup = if v2 { 2 * instances } else { instances };
        rows.push(builder.field(
            RowKind::LogUp,
            logup,
            None,
            Some((ProofOfWork::Seed, params.seed_pow)),
        ));

        let groups = sizes.claim_groups.iter().map(|&t| u128::from(t)).max();
        let openings = groups
            .unwrap_or(0)
            .max(u128::from(!sizes.matrices.is_empty()));
        rows.push(builder.field(RowKind::Openings, openings, None, None));

        for c in geometry {
            let (a, b) = (2 * u128::from(c.codeword), c.codeword_plus);
            let (per_query, bits) = exact::query_bits(a, b, c.queries, params.query_pow);
            let target = builder.target(true);
            rows.push(SoundnessRow {
                kind: RowKind::Queries(c.commitment),
                bound: ErrorBound::Queries {
                    codeword: c.codeword,
                    message: c.message,
                    mask: c.mask,
                    queries: c.queries,
                },
                theta: None,
                per_query: Some(per_query),
                bits: Some(bits),
                grinding: params.query_pow,
                covered_by: Some(ProofOfWork::Query),
                target,
                pass: target.map(|t| exact::queries_pass(a, b, c.queries, params.query_pow, t)),
            });
            let field = u128::from(c.theta) * u128::from(c.codeword);
            rows.push(builder.field(RowKind::Field(c.commitment), field, Some(c.theta), None));
        }

        // RFC-0014 'The link's combining challenge': ν_T for the evaluation, 1 for θ_T.
        let linking = sizes
            .matrices
            .iter()
            .map(|m| u128::from(m.variables) + 1)
            .max();
        rows.push(builder.field(RowKind::Linking, linking.unwrap_or(0), None, None));

        let target = builder.target(false);
        rows.push(SoundnessRow {
            kind: RowKind::Hash,
            bound: ErrorBound::Hash,
            theta: None,
            per_query: None,
            bits: Some(FixedBits(128 * UNIT)),
            grinding: 0,
            covered_by: None,
            target,
            pass: target.map(|t| t <= 128),
        });

        Ok(Self {
            params: *params,
            sizes: sizes.clone(),
            rows,
        })
    }

    /// The row with the least margin over its target, the first of any that tie, or with the
    /// fewest bits under a set held to no target; rows without a term are not counted. When
    /// every target is equal, as at `ni-100/1`, it is the row with the fewest bits.
    pub fn minimum(&self) -> Option<usize> {
        let key = |row: &SoundnessRow| match self.params.targets {
            Some(_) => row.margin(),
            None => row.bits.and_then(|b| i64::try_from(b.0).ok()),
        };
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(i, row)| key(row).map(|k| (k, i)))
            .min()
            .map(|(_, i)| i)
    }

    /// Whether every row reaches its target, or `None` under a set held to no target.
    pub fn meets_targets(&self) -> Option<bool> {
        self.params
            .targets
            .map(|_| self.rows.iter().all(|row| row.pass == Some(true)))
    }
}

/// The saturating sum.
fn sum(terms: impl Iterator<Item = u128>) -> u128 {
    terms.fold(0, u128::saturating_add)
}

/// The merge's degree (RFC-0054 'The merge').
const MERGE_DEGREE: u128 = 2;

/// A grouped stage's challenge errors beyond the ungrouped statement's: each level round's degree,
/// the merge's degree when it has a round, and its batching `P + G`.
fn group_challenges(groups: &LeafGroupSizes) -> impl Iterator<Item = u128> + '_ {
    groups
        .level_rounds
        .iter()
        .filter(|r| r.count > 0)
        .map(|r| u128::from(r.degree))
        .chain((groups.merge_rounds > 0).then_some(MERGE_DEGREE))
        .chain(Some(
            u128::from(groups.merge_claims).saturating_add(u128::from(groups.count)),
        ))
}

/// A grouped stage's union terms beyond the ungrouped statement's (RFC-0054 'Soundness'): `G`
/// more copies of the leaf stage's levels, `G·(Σ count·d + Σ_levels (m + 1))`, and the merge,
/// `2·n_max + P + G`.
fn group_union(sizes: &StatementSizes) -> u128 {
    let groups = &sizes.groups;
    let levels = sum(groups
        .level_rounds
        .iter()
        .map(|r| u128::from(r.count) * u128::from(r.degree)))
    .saturating_add(sum(sizes.tree_levels.iter().map(|&m| u128::from(m) + 1)));
    [
        u128::from(groups.count).saturating_mul(levels),
        MERGE_DEGREE * u128::from(groups.merge_rounds),
        u128::from(groups.merge_claims),
        u128::from(groups.count),
    ]
    .into_iter()
    .fold(0, u128::saturating_add)
}

/// Builds rows under the parameter set's targets.
struct Builder {
    targets: Option<Targets>,
}

impl Builder {
    /// The target of a row: `one_shot` for a row a proof of work covers, `bits` otherwise.
    fn target(&self, covered: bool) -> Option<u32> {
        self.targets
            .map(|t| if covered { t.one_shot } else { t.bits })
    }

    /// A field-form row, `ε = numerator/q`, under the proof of work `pow` and its bits.
    fn field(
        &self,
        kind: RowKind,
        numerator: u128,
        theta: Option<u32>,
        pow: Option<(ProofOfWork, u32)>,
    ) -> SoundnessRow {
        let grinding = pow.map_or(0, |(_, g)| g);
        let target = self.target(pow.is_some());
        SoundnessRow {
            kind,
            bound: ErrorBound::Field { numerator },
            theta,
            per_query: None,
            bits: exact::field_bits(numerator, grinding),
            grinding,
            covered_by: pow.map(|(p, _)| p),
            target,
            pass: target.map(|t| exact::field_pass(numerator, grinding, t)),
        }
    }
}

/// A commitment's proximity parameters.
struct Geometry {
    commitment: Commitment,
    codeword: u64,
    message: u64,
    mask: u64,
    queries: u32,
    theta: u32,
    /// `B = n + k + P`.
    codeword_plus: u128,
}

/// `⌈log2 x⌉` for `x ≥ 1`: the bits of an index below `x`.
fn ceil_log2(x: u128) -> u32 {
    u128::BITS - (x - 1).leading_zeros()
}

fn refuse<T>(rule: &'static str) -> Result<T, ProtocolError> {
    Err(ProtocolError::Sizes { rule })
}

/// Checks that the calculator can bound `params` at `sizes`, and gives each commitment's
/// proximity parameters: the advice's, then each matrix's in order.
fn validate(params: &ParamSet, sizes: &StatementSizes) -> Result<Vec<Geometry>, ProtocolError> {
    if !(1..=2).contains(&sizes.protocol) {
        return refuse("the statement's protocol is tessera-protocol/1 or tessera-protocol/2");
    }
    for queries in [params.advice_queries, params.weight_queries] {
        if !(1..=MAX_QUERIES).contains(&queries) {
            return refuse("τ_A and τ_W are 1 to 2^16");
        }
    }
    let advice = params.layout.advice;
    let advice_bits = u32::from(advice.log2_k) + u32::from(advice.log2_inv_rate);
    if advice_bits > 32 {
        return refuse("the advice codeword holds at most 2^32 symbols");
    }
    if sizes.advice_rows == 0 {
        return refuse("the advice spans a row");
    }
    if !sizes.matrices.windows(2).all(|w| w[0].matrix < w[1].matrix) {
        return refuse("the matrices are weight matrices in increasing order");
    }
    // RFC-0054 'Soundness': G is a power of two up to 2^14, 1 in tessera-protocol/2, and no
    // grouping has no grouped sizes.
    let groups = &sizes.groups;
    if !groups.count.is_power_of_two() || groups.count > LeafGroupSizes::MAX_COUNT {
        return refuse("the leaf groups are a power of two up to 2^14");
    }
    if groups.grouped() && sizes.protocol == 2 {
        return refuse("tessera-protocol/2 accepts no leaf groups");
    }
    if !groups.grouped()
        && (!groups.level_rounds.is_empty() || groups.merge_rounds > 0 || groups.merge_claims > 0)
    {
        return refuse("no leaf grouping has no level rounds and no merge");
    }
    let message = 1u64 << advice.log2_k;
    let mut geometry = vec![Geometry {
        commitment: Commitment::Advice,
        codeword: 1 << advice_bits,
        message,
        mask: u64::from(advice.mask_coeffs),
        queries: params.advice_queries,
        theta: ceil_log2(sizes.advice_rows.into()),
        codeword_plus: (1u128 << advice_bits)
            + u128::from(message)
            + u128::from(advice.mask_coeffs),
    }];
    let weights = params.layout.weights;
    let masked = sizes.protocol == 2;
    for m in &sizes.matrices {
        let codeword = m
            .width
            .checked_shl(u32::from(weights.log2_inv_rate))
            .filter(|n| n >> weights.log2_inv_rate == m.width);
        let Some(codeword) = codeword.filter(|&n| (1..=MAX_CODEWORD).contains(&n)) else {
            return refuse("every matrix's codeword holds 1 to 2^32 symbols");
        };
        // RFC-0049 'The calculator': tessera-protocol/1 commits no blinding row and no mask for
        // the weights, so a matrix counts its data rows alone, at the unmasked rate, and one
        // with no committed row has no proximity row. tessera-protocol/2 counts B_T and P_W.
        if masked && (m.claims == 0 || (m.matrix != WeightMatrix::ChannelParams && m.claims != 1)) {
            return refuse(
                "in tessera-protocol/2 an opened matrix has a claim, and one other than \
                 ChannelParams exactly one",
            );
        }
        let (rows, mask) = match masked {
            true => (
                u128::from(m.rows) + m.blinding_rows(weights.blinding_rows),
                u64::from(weights.mask_coeffs),
            ),
            false => (u128::from(m.rows), 0),
        };
        if rows == 0 {
            if masked {
                return refuse("every matrix holds a data or blinding row");
            }
            continue;
        }
        geometry.push(Geometry {
            commitment: Commitment::Weights(m.matrix),
            codeword,
            message: m.width,
            mask,
            queries: params.weight_queries,
            theta: ceil_log2(rows),
            codeword_plus: u128::from(codeword) + u128::from(m.width) + u128::from(mask),
        });
    }
    Ok(geometry)
}
