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
//! | [`Rounds`](RowKind::Rounds) | `D/q`, `D` the largest challenge error of the stages and tree levels | — | `bits` |
//! | [`Union`](RowKind::Union) | `U/q`, `U` the sum of them all | — | `bits` |
//! | [`Forests`](RowKind::Forests) | `s/q` for the largest forest of `2^s` trees | — | `bits` |
//! | [`LogUp`](RowKind::LogUp) | `(N + T)/q`, `2(N + T)/q` in `tessera-protocol/2`, the largest instance | `g_seed` | `one_shot` |
//! | [`Openings`](RowKind::Openings) | `T_g/q` for the largest claim group, `1/q` per weight matrix | — | `bits` |
//! | [`Queries`](RowKind::Queries) | `((n + k + P)/(2n))^τ` per commitment | `g_query` | `one_shot` |
//! | [`Field`](RowKind::Field) | `ϑ·n/q` per commitment | — | `bits` |
//! | [`Linking`](RowKind::Linking) | `ν/q` for the matrix of most variables | — | `bits` |
//! | [`Hash`](RowKind::Hash) | `2^{-128}`, BLAKE3's 256-bit output | — | `bits` |
//!
//! Here `q = |Fp2| = p^2`, and the targets are the parameter set's ([`Targets`]).

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
    /// `m_T`, its committed data rows. With the parameter set's `B_W` blinding rows they give
    /// `ϑ_T = ⌈log2(m_T + B_W)⌉`.
    pub rows: u64,
    /// `ν_T`, the variables of its multilinear polynomial, which the linking proof opens.
    pub variables: u32,
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
        /// `P`, each row's mask coefficients.
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
    /// no advice row, or a matrix with no data or blinding row; matrices out of
    /// [`WeightMatrix`] order.
    pub fn new(params: &ParamSet, sizes: &StatementSizes) -> Result<Self, ProtocolError> {
        let geometry = validate(params, sizes)?;
        let builder = Builder {
            targets: params.targets,
        };
        let v2 = sizes.protocol == 2;
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

        let linking = sizes.matrices.iter().map(|m| u128::from(m.variables)).max();
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
    for m in &sizes.matrices {
        let codeword = m
            .width
            .checked_shl(u32::from(weights.log2_inv_rate))
            .filter(|n| n >> weights.log2_inv_rate == m.width);
        let Some(codeword) = codeword.filter(|&n| (1..=MAX_CODEWORD).contains(&n)) else {
            return refuse("every matrix's codeword holds 1 to 2^32 symbols");
        };
        let rows = u128::from(m.rows) + u128::from(weights.blinding_rows);
        if rows == 0 {
            return refuse("every matrix holds a data or blinding row");
        }
        geometry.push(Geometry {
            commitment: Commitment::Weights(m.matrix),
            codeword,
            message: m.width,
            mask: u64::from(weights.mask_coeffs),
            queries: params.weight_queries,
            theta: ceil_log2(rows),
            codeword_plus: u128::from(codeword)
                + u128::from(m.width)
                + u128::from(weights.mask_coeffs),
        });
    }
    Ok(geometry)
}
