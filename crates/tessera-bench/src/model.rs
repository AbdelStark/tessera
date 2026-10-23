// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The cost model of 08 §1 (RFC-0020 'Executable cost model'), and the calculators that turn a
//! proof multiplier into proof time, a sampling rate and a detection probability (source §8,
//! RFC-0016 'Calculators').

use crate::error::{ModelError, Problem};

/// Seconds per picosecond.
const PS: f64 = 1e-12;

/// The prover's unit costs on one GPU (08 §2, RFC-0020 'Measurement protocol').
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitCosts {
    /// `c_c`: picoseconds per committed 64-bit element.
    pub commit_ps: f64,
    /// `c_ℓ`: picoseconds per LogUp-GKR lookup.
    pub lookup_ps: f64,
    /// `c_π`: picoseconds per product-tree leaf.
    pub product_leaf_ps: f64,
    /// The sumcheck glue on top of the commitment, lookup and tree work, as a fraction: `0.15`
    /// adds 15%.
    pub glue_fraction: f64,
    /// `W`: GPU-seconds of the pass over the weights that each proof batch pays once.
    pub weight_pass_gpu_s: f64,
    /// `R`: native INT8 operations per second on one GPU.
    pub native_ops_per_s: f64,
    /// Picoseconds per leaf of one streamed round over the table trees, which recompute their
    /// leaves from the tables' rows and the stored multiplicities and regenerate nothing
    /// (RFC-0054 'The table batch'). RFC-0018 estimated 6 to 8 ps per leaf for a streamed pass.
    pub streaming_pass_ps: f64,
    /// Picoseconds per merged tensor element of a grouped leaf stage: its fold into the merge's
    /// tables while its group is resident, and its fold in the merge's pass (RFC-0054 'The
    /// merge').
    pub merge_ps: f64,
    /// Picoseconds per committed element to evaluate its row's masked codeword at the queried
    /// columns, hashing nothing: the column answers' pass (RFC-0018 P5).
    pub column_answer_ps: f64,
    /// Picoseconds per committed element to fold it into the opening's combined rows: `u_ξ`, and
    /// the `u_g` of each claim group on its tensor, one product of an `Fp2` coefficient by the
    /// element per row (RFC-0011 step 4). This is the opening pass's arithmetic, beside its
    /// regeneration (RFC-0018 P4).
    pub row_combination_ps: f64,
}

impl UnitCosts {
    /// The source's assumed unit costs on an H100 SXM (source §5): 30, 40 and 12 ps, 15% glue,
    /// one 0.24 GPU-s weight pass per batch, native INT8 at 60% of 1,979 TOPS. The source has no
    /// streamed rounds, no merge, no column-answer pass and no row combination, so they cost
    /// nothing here; RFC-0018 estimates 6 to 8 ps per leaf for a streamed round, and RFC-0054 2 to
    /// 4 ps per element for the merge.
    pub const SOURCE: Self = Self {
        commit_ps: 30.0,
        lookup_ps: 40.0,
        product_leaf_ps: 12.0,
        glue_fraction: 0.15,
        weight_pass_gpu_s: 0.24,
        native_ops_per_s: 0.6 * 1979e12,
        streaming_pass_ps: 0.0,
        merge_ps: 0.0,
        column_answer_ps: 0.0,
        row_combination_ps: 0.0,
    };
}

/// What a model costs per token: the compiler's commitment report, or the source's counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Counts {
    /// `C_lin`: committed elements per token in the linear layers.
    pub commits_per_token: f64,
    /// `K_lin`: lookups per token in the linear layers.
    pub lookups_per_token: f64,
    /// `A`: attention scores per token per unit of context, heads × layers / 2.
    pub scores_per_token_per_ctx: f64,
    /// `F_lin`: native INT8 operations per token in the linear layers.
    pub native_ops_lin: f64,
    /// `F_att`: native INT8 operations per token per unit of context in attention.
    pub native_ops_att_per_ctx: f64,
    /// `n_regen`: witness regenerations, each about one native forward pass (RISK-9).
    pub regen_full_equivalents: f64,
    /// `M`: table rows a proof commits a multiplicity for and builds its fraction tree over, all
    /// its tables together: the commitment report's multiplicities (RFC-0008).
    pub table_rows: f64,
    /// `b`: the score block of RFC-0034. At context `S` a token commits `A·(S + min(S, b))`
    /// scores: the valid half, and the masked halves of the blocks on the diagonal, which the
    /// advice layout cannot skip. Its lookups and tree leaves stay `A·S`: the masked ones are
    /// constant subtrees. A block of 0 commits the valid half alone.
    pub score_block: f64,
}

impl Counts {
    /// Llama-3.1-405B as the source counts it (source §5), the vectors of
    /// `spec/vectors/cost-model/` giving each figure's line:
    ///
    /// - 137,216 committed elements and lookups per layer per token, over 126 layers;
    /// - `A = 128 · 126 / 2 = 8,064` scores per token per unit of context, 512 native operations
    ///   each;
    /// - `F_lin = 2 · 405,853,388,800`, two operations per parameter: the source writes
    ///   `8.1×10^11`, and this is the reading that reproduces its chart;
    /// - no regeneration, which the source's model leaves out (RISK-9);
    /// - no table rows, which it leaves out too;
    /// - no score block: the source commits the valid half of the scores alone.
    pub const SOURCE_405B: Self = Self {
        commits_per_token: 137_216.0 * 126.0,
        lookups_per_token: 137_216.0 * 126.0,
        scores_per_token_per_ctx: 8_064.0,
        native_ops_lin: 2.0 * 405_853_388_800.0,
        native_ops_att_per_ctx: 512.0 * 8_064.0,
        regen_full_equivalents: 0.0,
        table_rows: 0.0,
        score_block: 0.0,
    };
}

/// The memory the leaf stage's trees must fit in (RISK-10, RFC-0054 'When groups apply').
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryModel {
    /// Device memory left for the leaf stage's trees, in bytes.
    pub free_bytes: f64,
    /// Bytes per leaf of a level held in memory.
    pub bytes_per_leaf: f64,
}

impl MemoryModel {
    /// One 8×H100 node at 405B (RFC-0018 'Leaf levels'): about 0.2 TB of HBM left after the
    /// weights, and 32 bytes per leaf, two `Fp2` tables.
    pub const NODE_405B: Self = Self {
        free_bytes: 0.2e12,
        bytes_per_leaf: 32.0,
    };
}

/// The proof multiplier `μ` at one context, term by term, each in units of the native time of a
/// token at that context (08 §1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuBreakdown {
    /// Commitments and lookups of the linear layers, with glue.
    pub linear: f64,
    /// Commitments, lookups and product-tree leaves of the attention scores, with glue.
    pub attention: f64,
    /// The weight pass, shared by the batch's tokens.
    pub weights: f64,
    /// Witness regenerations (RISK-9): `n_regen`, and a grouped leaf stage's merge pass
    /// (RFC-0054).
    pub regen: f64,
    /// The leaf stage's memory plan beside its regenerations (RISK-10, RFC-0054): a grouped
    /// stage's merge folds, and the table trees' streamed rounds.
    pub streaming: f64,
    /// The column answers' evaluation of every committed element (RFC-0018 P5).
    pub column_answers: f64,
    /// The opening pass's combination of every committed element into the combined rows
    /// (RFC-0018 P4).
    pub row_combination: f64,
    /// The table side of LogUp: each table row's multiplicity and fraction-tree leaf, shared by
    /// the batch's tokens (RFC-0008).
    pub tables: f64,
    /// The sum of the terms.
    pub total: f64,
}

fn input(name: &'static str, value: f64, positive: bool) -> Result<(), ModelError> {
    let problem = if !value.is_finite() {
        Some(Problem::NotFinite)
    } else if positive && value <= 0.0 {
        Some(Problem::NotPositive)
    } else if value < 0.0 {
        Some(Problem::Negative)
    } else {
        None
    };
    match problem {
        Some(problem) => Err(ModelError::Input { name, problem }),
        None => Ok(()),
    }
}

fn output(name: &'static str, value: f64) -> Result<f64, ModelError> {
    match value.is_finite() {
        true => Ok(value),
        false => Err(ModelError::Output { name }),
    }
}

impl UnitCosts {
    /// Every cost finite and non-negative, the native rate positive.
    fn check(&self) -> Result<(), ModelError> {
        input("commit_ps", self.commit_ps, false)?;
        input("lookup_ps", self.lookup_ps, false)?;
        input("product_leaf_ps", self.product_leaf_ps, false)?;
        input("glue_fraction", self.glue_fraction, false)?;
        input("weight_pass_gpu_s", self.weight_pass_gpu_s, false)?;
        input("native_ops_per_s", self.native_ops_per_s, true)?;
        input("streaming_pass_ps", self.streaming_pass_ps, false)?;
        input("merge_ps", self.merge_ps, false)?;
        input("column_answer_ps", self.column_answer_ps, false)?;
        input("row_combination_ps", self.row_combination_ps, false)
    }
}

impl Counts {
    /// Every count finite and non-negative.
    fn check(&self) -> Result<(), ModelError> {
        input("commits_per_token", self.commits_per_token, false)?;
        input("lookups_per_token", self.lookups_per_token, false)?;
        input(
            "scores_per_token_per_ctx",
            self.scores_per_token_per_ctx,
            false,
        )?;
        input("native_ops_lin", self.native_ops_lin, false)?;
        input("native_ops_att_per_ctx", self.native_ops_att_per_ctx, false)?;
        input("regen_full_equivalents", self.regen_full_equivalents, false)?;
        input("table_rows", self.table_rows, false)?;
        input("score_block", self.score_block, false)
    }

    /// The leaves a token adds to the leaf stage's lookup trees and product forests at
    /// `context`: its `K` lookups, and per score an `Exp` lookup and a `SoftmaxRow` leaf,
    /// `K + 2·A·S`. Masked scores are constant subtrees (RFC-0034) and add none.
    fn leaves_per_token(&self, context: u32) -> f64 {
        self.lookups_per_token + 2.0 * self.scores_per_token_per_ctx * f64::from(context)
    }

    /// Native seconds per token at `context` on one GPU, `(F_lin + F_att·S)/R`, positive.
    fn native_s(&self, u: &UnitCosts, context: u32) -> Result<f64, ModelError> {
        let ops = self.native_ops_lin + self.native_ops_att_per_ctx * f64::from(context);
        let native = output("native_s", ops / u.native_ops_per_s)?;
        match native > 0.0 {
            true => Ok(native),
            // No native work at this context: nothing to compare the proof with.
            false => Err(ModelError::Input {
                name: "native_ops_lin",
                problem: Problem::NotPositive,
            }),
        }
    }
}

impl MemoryModel {
    fn check(&self) -> Result<(), ModelError> {
        input("free_bytes", self.free_bytes, true)?;
        input("bytes_per_leaf", self.bytes_per_leaf, false)
    }
}

/// The most token groups a leaf stage takes, `2^14` (RFC-0054 'The group count'): a group of one
/// token in a 16K-token batch, and few enough that RFC-0014's union bound keeps 100 bits.
pub const MAX_LEAF_GROUPS: u32 = 1 << 14;

/// How a batch's leaf stage meets the memory budget (RFC-0054 'When groups apply').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafPlan {
    /// `G`: the token groups, 1 while the lookup trees and product forests fit in memory.
    pub groups: u32,
    /// The streamed rounds of the batch that holds the table trees, summed over its levels. Each
    /// recomputes the table trees' leaves from the rows and the stored multiplicities, and
    /// regenerates nothing.
    pub table_rounds: u32,
}

impl LeafPlan {
    /// The regenerations the plan adds to P2's one (RFC-0018 'Pass structure'): the merge's pass,
    /// once, when the stage is grouped.
    pub fn merge_passes(&self) -> u32 {
        u32::from(self.groups > 1)
    }
}

/// The bytes per leaf a leaf stage keeps beside its level tables: the tensor value its leaf reads,
/// one integer, which stays until the stage's leaf point is known and the tensors' values at it
/// are computed (RFC-0045 'The split', RFC-0054 'When groups apply').
pub const TENSOR_BYTES_PER_LEAF: f64 = 8.0;

/// The rounds a batch of `bytes` streams over every level when `free` holds less: the leaf
/// level `c = ⌈log2(bytes/free)⌉` of them, the level above `c − 1`, and so on, `c(c + 1)/2` in
/// all. The count is exact: halving is exact in floating point.
fn streamed_rounds(bytes: f64, free: f64) -> u32 {
    let (mut level, mut c) = (bytes, 0u32);
    // A finite level reaches `free` in at most about 2,100 halvings.
    while level > free {
        level /= 2.0;
        c += 1;
    }
    c * (c + 1) / 2
}

/// The leaf stage's plan for a batch of `batch_tokens` tokens at `context`, from its sizes
/// (RFC-0054 'When groups apply'). Its lookup trees and product forests hold
/// `T·(K + 2·A·S)·(bytes_per_leaf + TENSOR_BYTES_PER_LEAF)` bytes, their level tables and the
/// tensors their leaves read, and its table trees `M·bytes_per_leaf`:
///
/// - while the lookup trees and forests fit in `free_bytes`, the stage is one batch, and its
///   levels that do not fit with the table trees stream the table trees alone;
/// - otherwise the stage takes the fewest groups `G = 2^g` whose lookup trees and forests fit,
///   one share of the tokens each, and the table trees stream in a batch of their own.
///
/// The planner splits `G` over interleaved token subsets of at least 256 tokens, which keep a
/// group's GEMMs compute-bound, and over layer subsets; the plan counts `G` alone.
///
/// # Errors
///
/// [`ModelError::Input`] for a count or a memory figure that is not finite, a negative one, or
/// no free memory; `free_bytes` [`Problem::TooSmall`] when [`MAX_LEAF_GROUPS`] groups, or one
/// token's leaves, do not fit.
pub fn leaf_plan(
    counts: &Counts,
    mem: &MemoryModel,
    context: u32,
    batch_tokens: u32,
) -> Result<LeafPlan, ModelError> {
    counts.check()?;
    mem.check()?;
    let leaves = counts.leaves_per_token(context) * f64::from(batch_tokens);
    let per_leaf = mem.bytes_per_leaf + TENSOR_BYTES_PER_LEAF;
    let trees = output("leaf_stage_bytes", leaves * per_leaf)?;
    let tables = output("table_tree_bytes", counts.table_rows * mem.bytes_per_leaf)?;
    let too_small = ModelError::Input {
        name: "free_bytes",
        problem: Problem::TooSmall,
    };
    let (mut groups, mut group) = (1u32, trees);
    while group > mem.free_bytes {
        let fewer_tokens = f64::from(groups) >= f64::from(batch_tokens);
        if groups >= MAX_LEAF_GROUPS || fewer_tokens {
            return Err(too_small);
        }
        groups *= 2;
        group /= 2.0;
    }
    let table_rounds = match groups {
        1 => streamed_rounds(trees + tables, mem.free_bytes),
        _ => streamed_rounds(tables, mem.free_bytes),
    };
    Ok(LeafPlan {
        groups,
        table_rounds,
    })
}

/// The streamed passes the leaf stage of a batch of `batch_tokens` tokens at `context` would take
/// in one batch with shared challenges, the design RFC-0054 replaces (its 'Motivation'). Every
/// level of every tree streams until it fits with the others, one regeneration pass per round:
/// `c(c + 1)/2`, with `c = ⌈log2(N·bytes_per_leaf/free_bytes)⌉` for the
/// `N = T·(K + 2·A·S) + M` leaves of the lookup trees, the product forests and the table
/// trees.
///
/// # Errors
///
/// [`ModelError::Input`] for a count or a memory figure that is not finite, a negative one, or
/// no free memory.
pub fn shared_streaming_passes(
    counts: &Counts,
    mem: &MemoryModel,
    context: u32,
    batch_tokens: u32,
) -> Result<u32, ModelError> {
    counts.check()?;
    mem.check()?;
    let leaves = counts.leaves_per_token(context) * f64::from(batch_tokens) + counts.table_rows;
    let bytes = output("leaf_stage_bytes", leaves * mem.bytes_per_leaf)?;
    Ok(streamed_rounds(bytes, mem.free_bytes))
}

/// `μ` at `context` tokens of context for a batch of `batch_tokens` tokens, term by term
/// (08 §1, RFC-0020 'Executable cost model'):
///
/// ```text
/// μ(S) = [(1 + glue)·(C·c_c + K·c_ℓ) + (1 + glue)·A·(S'·c_c + S·(c_ℓ + c_π)) + W/T
///         + p_m·c_m·(K + 2·A·S) + r_t·c_s·M/T
///         + (C + A·S')·(c_a + c_r) + (1 + glue)·M·(c_c + c_ℓ)/T]
///        / [F/R]  +  n_regen + p_m,        S' = S + min(S, b),  F = F_lin + F_att·S
/// ```
///
/// with `T` the batch's tokens, and the leaf stage's plan from [`leaf_plan`]: `p_m` its merge
/// passes, one when it is grouped, each a regeneration with `c_m` per merged element, about one
/// per leaf; and `r_t` the table trees' streamed rounds at `c_s` per table leaf. The merge's
/// regeneration counts only when `n_regen` is positive: a model that counts no regeneration, as
/// the source's, counts none. `c_a` and `c_r`
/// are the column answers' and the row combination's costs per committed element. A token
/// commits `C` elements in its linear layers and `A·S'` scores, `b` the score block (RFC-0034),
/// and looks up and multiplies `A·S` of them. Each of the `M` table rows costs one committed
/// multiplicity and one fraction-tree leaf, priced as a lookup, once per proof. The source's
/// counts and unit costs regenerate nothing, merge and stream nothing at no cost, price neither
/// the column answers nor the row combination, and count no table rows and no score block. With
/// them, this is the source's formula (source §5).
///
/// # Errors
///
/// [`ModelError::Input`] for an input that is not finite, a negative one, a native rate or free
/// memory that is not positive, memory in which no plan fits, an empty batch, or no native work
/// to compare with; [`ModelError::Output`] for a term that overflows.
///
/// ```
/// use tessera_bench::{Counts, MemoryModel, UnitCosts, mu};
///
/// let mu_16k = mu(&Counts::SOURCE_405B, &UnitCosts::SOURCE, &MemoryModel::NODE_405B, 16_384, 16_384)?;
/// // The source charts 18.75 at 16K (source §5).
/// assert!((mu_16k.total - 18.75).abs() < 0.05);
/// # Ok::<(), tessera_bench::ModelError>(())
/// ```
pub fn mu(
    counts: &Counts,
    u: &UnitCosts,
    mem: &MemoryModel,
    context: u32,
    batch_tokens: u32,
) -> Result<MuBreakdown, ModelError> {
    u.check()?;
    if batch_tokens == 0 {
        return Err(ModelError::Input {
            name: "batch_tokens",
            problem: Problem::NotPositive,
        });
    }
    let plan = leaf_plan(counts, mem, context, batch_tokens)?;
    let native = counts.native_s(u, context)?;
    let s = f64::from(context);
    let glue = 1.0 + u.glue_fraction;
    let linear =
        glue * (counts.commits_per_token * u.commit_ps + counts.lookups_per_token * u.lookup_ps);
    // The diagonal blocks' masked halves: committed, never looked up or multiplied (RFC-0034).
    let diagonal = counts.scores_per_token_per_ctx * s.min(counts.score_block);
    let per_score = u.commit_ps + u.lookup_ps + u.product_leaf_ps;
    let attention =
        glue * counts.scores_per_token_per_ctx * s * per_score + glue * diagonal * u.commit_ps;
    let weights = u.weight_pass_gpu_s / f64::from(batch_tokens);
    // A grouped stage's merge folds an element per leaf, and the table trees' streamed rounds
    // recompute their leaves, shared by the batch's tokens (RFC-0054).
    let merge = f64::from(plan.merge_passes()) * u.merge_ps * counts.leaves_per_token(context);
    let table_streaming = f64::from(plan.table_rounds) * u.streaming_pass_ps * counts.table_rows
        / f64::from(batch_tokens);
    let streaming = (merge + table_streaming) * PS;
    let committed = counts.commits_per_token + counts.scores_per_token_per_ctx * s + diagonal;
    let column_answers = committed * u.column_answer_ps;
    let row_combination = committed * u.row_combination_ps;
    let tables = glue * counts.table_rows * (u.commit_ps + u.lookup_ps) / f64::from(batch_tokens);
    let linear = output("linear", linear * PS / native)?;
    let attention = output("attention", attention * PS / native)?;
    let weights = output("weights", weights / native)?;
    let streaming = output("streaming", streaming / native)?;
    let column_answers = output("column_answers", column_answers * PS / native)?;
    let row_combination = output("row_combination", row_combination * PS / native)?;
    let tables = output("tables", tables * PS / native)?;
    // The merge's pass is one more regeneration, counted when the model counts regenerations:
    // the source's counts count none (RISK-9, RFC-0054).
    let merge_regen = match counts.regen_full_equivalents > 0.0 {
        true => f64::from(plan.merge_passes()),
        false => 0.0,
    };
    let regen = counts.regen_full_equivalents + merge_regen;
    let total = linear
        + attention
        + weights
        + regen
        + streaming
        + column_answers
        + row_combination
        + tables;
    let total = output("total", total)?;
    Ok(MuBreakdown {
        linear,
        attention,
        weights,
        regen,
        streaming,
        column_answers,
        row_combination,
        tables,
        total,
    })
}

/// Seconds to prove a request of `context` tokens on `gpus` GPUs at multiplier `mu`: `μ` times
/// the native time of its tokens, `S·(F_lin + F_att·S)/(G·R)`, every token at the full context as
/// the source counts it (source §8).
///
/// # Errors
///
/// [`ModelError::Input`] for counts or unit costs [`mu`] refuses, a multiplier that is not
/// finite or negative, or no GPU; [`ModelError::Output`] for a time that overflows.
pub fn proof_time_s(
    mu: f64,
    counts: &Counts,
    u: &UnitCosts,
    context: u32,
    gpus: u32,
) -> Result<f64, ModelError> {
    input("mu", mu, false)?;
    u.check()?;
    counts.check()?;
    if gpus == 0 {
        return Err(ModelError::Input {
            name: "gpus",
            problem: Problem::NotPositive,
        });
    }
    let native = counts.native_s(u, context)?;
    output(
        "proof_time_s",
        mu * f64::from(context) * native / f64::from(gpus),
    )
}

/// The sampling rate that keeps the amortized overhead at `target_overhead` when proofs cost
/// `mu` times the native prefill and prefill is `eta` of serving time per token:
/// `s = target/(μ·η)`, and every request when that exceeds 1 (source §8; RFC-0016
/// 'Calculators').
///
/// # Errors
///
/// [`ModelError::Input`] for a target that is not finite or is negative, or a multiplier or
/// `η` that is not positive.
pub fn sampling_rate(target_overhead: f64, mu: f64, eta: f64) -> Result<f64, ModelError> {
    input("target_overhead", target_overhead, false)?;
    input("mu", mu, true)?;
    input("eta", eta, true)?;
    Ok(output("sampling_rate", target_overhead / (mu * eta))?.min(1.0))
}

/// The probability that a sample at `rate` selects at least one of `violations` violating
/// requests, `1 − (1 − s)^v`: each request is committed before the sample is drawn and selected
/// independently (source §8; RFC-0016 'Calculators').
///
/// # Errors
///
/// [`ModelError::Input`] for a rate that is not finite or outside `[0, 1]`.
pub fn detection_probability(rate: f64, violations: u32) -> Result<f64, ModelError> {
    input("rate", rate, false)?;
    if rate > 1.0 {
        return Err(ModelError::Input {
            name: "rate",
            problem: Problem::AboveOne,
        });
    }
    if violations == 0 {
        return Ok(0.0);
    }
    // 1 − exp(v·ln(1 − s)), accurate for small rates and for many violations; a rate of 1 gives
    // exp(−∞) = 0 and so 1.
    Ok(-(f64::from(violations) * (-rate).ln_1p()).exp_m1())
}
