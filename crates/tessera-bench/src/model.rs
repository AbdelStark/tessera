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
    /// Picoseconds per attention score of one streamed pass over a tree's leaf level (RISK-10,
    /// RFC-0018 'Leaf levels').
    pub streaming_pass_ps: f64,
    /// Picoseconds per committed element to evaluate its row's masked codeword at the queried
    /// columns, hashing nothing: the column answers' pass (RFC-0018 P5).
    pub column_answer_ps: f64,
}

impl UnitCosts {
    /// The source's assumed unit costs on an H100 SXM (source §5): 30, 40 and 12 ps, 15% glue,
    /// one 0.24 GPU-s weight pass per batch, native INT8 at 60% of 1,979 TOPS. The source has no
    /// streamed passes and no column-answer pass, so both cost nothing here; RFC-0018 estimates 6
    /// to 8 ps per score for a streamed pass.
    pub const SOURCE: Self = Self {
        commit_ps: 30.0,
        lookup_ps: 40.0,
        product_leaf_ps: 12.0,
        glue_fraction: 0.15,
        weight_pass_gpu_s: 0.24,
        native_ops_per_s: 0.6 * 1979e12,
        streaming_pass_ps: 0.0,
        column_answer_ps: 0.0,
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
    /// - no regeneration, which the source's model leaves out (RISK-9).
    pub const SOURCE_405B: Self = Self {
        commits_per_token: 137_216.0 * 126.0,
        lookups_per_token: 137_216.0 * 126.0,
        scores_per_token_per_ctx: 8_064.0,
        native_ops_lin: 2.0 * 405_853_388_800.0,
        native_ops_att_per_ctx: 512.0 * 8_064.0,
        regen_full_equivalents: 0.0,
    };
}

/// The memory a tree's leaf level must fit in before its rounds stop streaming (RISK-10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryModel {
    /// Device memory left for a leaf level, in bytes.
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
    /// Witness regenerations (RISK-9).
    pub regen: f64,
    /// Streamed passes over the attention leaves (RISK-10).
    pub streaming: f64,
    /// The column answers' evaluation of every committed element (RFC-0018 P5).
    pub column_answers: f64,
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
        input("column_answer_ps", self.column_answer_ps, false)
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
        input("regen_full_equivalents", self.regen_full_equivalents, false)
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

/// `c(S)`: the streamed passes over the attention leaves of a batch of `batch_tokens` tokens at
/// `context`, `max(0, ⌈log2(N_level · bytes_per_leaf / free_bytes)⌉)` with
/// `N_level = A·S·T` leaves (RFC-0020 'Executable cost model', RFC-0018 'Leaf levels'): one
/// pass per round until the folded level, halving every round, fits in memory.
///
/// The count is exact: the level's size halves in floating point without rounding, so a level
/// of exactly `2^c` times the memory takes `c` passes.
///
/// # Errors
///
/// [`ModelError::Input`] for a count or a memory figure that is not finite, a negative one, or
/// no free memory.
pub fn streaming_passes(
    counts: &Counts,
    mem: &MemoryModel,
    context: u32,
    batch_tokens: u32,
) -> Result<u32, ModelError> {
    counts.check()?;
    mem.check()?;
    let leaves = counts.scores_per_token_per_ctx * f64::from(context) * f64::from(batch_tokens);
    let mut level = output("leaf_level_bytes", leaves * mem.bytes_per_leaf)?;
    let mut passes = 0;
    // Each halving is exact, and a finite level reaches `free_bytes` in at most about 2,100.
    while level > mem.free_bytes {
        level /= 2.0;
        passes += 1;
    }
    Ok(passes)
}

/// `μ` at `context` tokens of context for a batch of `batch_tokens` tokens, term by term
/// (08 §1, RFC-0020 'Executable cost model'):
///
/// ```text
/// μ(S) = [(1 + glue)·(C·c_c + K·c_ℓ) + (1 + glue)·A·S·(c_c + c_ℓ + c_π) + W/T
///         + c(S)·c_s·A·S + (C + A·S)·c_a] / [(F_lin + F_att·S)/R]  +  n_regen
/// ```
///
/// with `c_s` the streamed pass's cost per score, [`streaming_passes`] its count `c(S)`, and
/// `c_a` the column answers' cost per committed element, of which a token commits `C` in its
/// linear layers and one per attention score. With the source's counts and unit costs, which
/// regenerate nothing, stream nothing and price no column answers, this is the source's formula
/// (source §5).
///
/// # Errors
///
/// [`ModelError::Input`] for an input that is not finite, a negative one, a native rate or free
/// memory that is not positive, an empty batch, or no native work to compare with;
/// [`ModelError::Output`] for a term that overflows.
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
    let passes = streaming_passes(counts, mem, context, batch_tokens)?;
    let native = counts.native_s(u, context)?;
    let s = f64::from(context);
    let glue = 1.0 + u.glue_fraction;
    let linear =
        glue * (counts.commits_per_token * u.commit_ps + counts.lookups_per_token * u.lookup_ps);
    let per_score = u.commit_ps + u.lookup_ps + u.product_leaf_ps;
    let attention = glue * counts.scores_per_token_per_ctx * s * per_score;
    let weights = u.weight_pass_gpu_s / f64::from(batch_tokens);
    let streaming = f64::from(passes) * u.streaming_pass_ps * counts.scores_per_token_per_ctx * s;
    let committed = counts.commits_per_token + counts.scores_per_token_per_ctx * s;
    let column_answers = committed * u.column_answer_ps;
    let linear = output("linear", linear * PS / native)?;
    let attention = output("attention", attention * PS / native)?;
    let weights = output("weights", weights / native)?;
    let streaming = output("streaming", streaming * PS / native)?;
    let column_answers = output("column_answers", column_answers * PS / native)?;
    let regen = counts.regen_full_equivalents;
    let total = linear + attention + weights + regen + streaming + column_answers;
    let total = output("total", total)?;
    Ok(MuBreakdown {
        linear,
        attention,
        weights,
        regen,
        streaming,
        column_answers,
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
