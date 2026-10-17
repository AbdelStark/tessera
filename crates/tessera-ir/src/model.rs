// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The integer model `D` and its quantization constants (03 §2, RFC-0005, RFC-0007).
//!
//! Every value is an integer or a rational (INV-DM-6), and every type encodes in TCE
//! (RFC-0003): structs field by field, enums by declaration index. [`IntegerModel`] is the
//! artifact `0x0101` that the Python front-end writes and the Rust tools read. Its
//! [`QuantSpec`] is the claim list's [`PublicQuantSpec`], field for field, beside the hidden
//! per-channel classes that RFC-0012 commits with the weights.
//!
//! Weights and channel classes are secret-class (05 §3): the debug forms of [`IntegerModel`],
//! [`WeightTensor`], [`QuantSpec`] and [`ChannelParams`] show shapes, never an entry.

use core::fmt;

use tessera_codec::{Ascii, Decode, Encode, Map, Rational};

use crate::WeightType;

/// 32 bytes of BLAKE3.
pub type Digest = [u8; 32];

/// The model family: Llama-3 in v1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum ArchFamily {
    /// Llama-3 and Llama-3.1.
    LlamaV3,
}

/// Llama-3.1's RoPE: the integer parameters of RFC-0008's `Rope` table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct RopeSpec {
    /// The base `θ`.
    pub theta: u32,
    /// Llama-3's scaling `factor`.
    pub factor: u16,
    /// `low_freq_factor`.
    pub low_freq_factor: u16,
    /// `high_freq_factor`.
    pub high_freq_factor: u16,
    /// `original_max_position_embeddings`.
    pub original_max: u32,
    /// The constants' precision: 14 in `D`.
    pub bits: u8,
}

/// The architecture of `D` (03 §2).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct ArchSpec {
    /// The family.
    pub family: ArchFamily,
    /// The decoder layers.
    pub n_layers: u32,
    /// The residual width.
    pub d_model: u32,
    /// The query heads.
    pub n_heads: u32,
    /// The key and value heads.
    pub n_kv_heads: u32,
    /// The head width.
    pub d_head: u32,
    /// The MLP's inner width.
    pub d_ff: u32,
    /// The vocabulary size.
    pub vocab: u32,
    /// The longest context.
    pub max_context: u32,
    /// RoPE.
    pub rope: RopeSpec,
    /// RMSNorm's `ε`, exactly.
    pub rms_eps: Rational,
}

/// The dataflow of `D` (RFC-0005 'Profiles').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum Profile {
    /// Every matmul operand INT8, every accumulator INT32: exact mode is available.
    Servable,
    /// The source's commitment-minimal dataflow.
    Minimal,
}

/// A rescale of a layer, in the order of RFC-0005 'Integer semantics'.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Encode, Decode)]
pub enum Site {
    /// The attention norm's output, `y = rq(res · ρ, r_n)`.
    AttnNorm,
    /// The queries after RoPE; their classes follow `W_Q`'s output channels.
    RopeQ,
    /// The keys after RoPE; their classes follow `W_K`'s output channels.
    RopeK,
    /// The values, `v = rq(acc, r_v,c)`; classes from `W_V`.
    V,
    /// The attention output, `o8 = rq(o, r_o)`; `Servable` only.
    AttnOut,
    /// The O projection's accumulator into the residual; classes from `W_O`.
    AttnResidual,
    /// The MLP norm's output.
    MlpNorm,
    /// SiLU of the gate's accumulator, through the `Silu` table; classes from `W_Gate`.
    Silu,
    /// The up projection, `u8 = rq(u, r_u)`; `Servable` only; classes from `W_Up`.
    Up,
    /// The product, `h = rq(a · u8, r_h)`; `Servable` only.
    Product,
    /// The down projection's accumulator into the residual; classes from `W_Down`.
    MlpResidual,
}

/// A quantized or rescaled tensor of `D`, ordered as declared: the embedding, the layers in
/// order with their sites in [`Site`] order, the final norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Encode, Decode)]
pub enum QuantPoint {
    /// The INT8 embedding rows `x0`.
    Embed,
    /// A rescale of a layer.
    Layer {
        /// The layer.
        layer: u32,
        /// The rescale.
        site: Site,
    },
    /// The final norm's output, the LM head's input.
    FinalNorm,
}

/// The quantization data the claim list reveals (RFC-0007 'Schema').
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct PublicQuantSpec {
    /// The exponent `e` of each INT8 tensor's scale `2^e`, sorted by point.
    pub act_exponents: Vec<(QuantPoint, i8)>,
    /// The residual stream's exponent.
    pub residual_exponent: i8,
    /// The residual stream's width: 20.
    pub residual_bits: u8,
    /// `κ` of the reciprocal square root: the largest, at most 31, with `⌊d · 2^{2κ} / e⌋ < 2^64`
    /// (RFC-0023).
    pub rsqrt_kappa: u8,
    /// `e`, the exact integer form of `d · ε / σ_res^2`, rounded up.
    pub rms_eps_int: u64,
    /// `σ` of the softmax exponentials: 16.
    pub exp_sigma: u8,
    /// Each layer's `σ_q σ_k / √d_head`, exactly.
    pub score_scales: Vec<(u32, Rational)>,
    /// Each `pot` rescale site's class shifts `r`, 1 to 63, at most four: its `Requant` table's
    /// `r_by_class`. A site after a weight has one class per weight exponent class, every other
    /// site one.
    pub shift_class_sets: Vec<(QuantPoint, Vec<i8>)>,
    /// Each `affine` rescale site's shift `r`, 1 to 63, sorted by point (RFC-0025). Its key is
    /// the product of what a `pot` site would rescale and the hidden 16-bit multipliers of the
    /// weight before it. A rescale site is in this list or in `shift_class_sets`, never both.
    pub affine_sites: Vec<(QuantPoint, i8)>,
    /// RoPE.
    pub rope: RopeSpec,
    /// The logits' width: 28.
    pub logits_bits: u8,
}

/// A weight's hidden per-output-channel parameters, committed with the weights as the
/// `ChannelParams` matrix (RFC-0012 'Layout').
#[derive(Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub struct ChannelParams {
    /// The weight type.
    pub ty: WeightType,
    /// The layer, 0 for `Embed` and `LmHead`.
    pub layer: u32,
    /// Each output channel's index into its site's shift classes; empty at an affine site.
    pub class: Vec<u8>,
    /// Each output channel's multiplier `M_c`, from 1 to `2^16 − 1`, at an affine site; empty at
    /// a `pot` site (RFC-0025).
    pub multiplier: Vec<u16>,
}

/// Shows the weight and the channel count, never a class or a multiplier (05 §3).
impl fmt::Debug for ChannelParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChannelParams")
            .field("ty", &self.ty)
            .field("layer", &self.layer)
            // One class per channel at a `pot` site, one multiplier at an affine one (RFC-0025).
            .field("channels", &self.class.len().max(self.multiplier.len()))
            .field("affine", &!self.multiplier.is_empty())
            .finish()
    }
}

/// Every quantization constant of `D` (03 §2 'Quantization').
#[derive(Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub struct QuantSpec {
    /// The claim list's copy.
    pub public: PublicQuantSpec,
    /// Each layer's `Silu` table input exponent `in_exp`, sorted by layer: public, stated by the
    /// table's `TableSpec` in the claim list, but not derivable from the rest.
    pub silu_in_exponents: Vec<(u32, i8)>,
    /// The hidden part: one per weight whose outputs a site of several classes, or an affine
    /// site, rescales, sorted by `(ty, layer)`.
    pub channels: Vec<ChannelParams>,
}

/// Shows the public part and the hidden part's shapes (05 §3).
impl fmt::Debug for QuantSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuantSpec")
            .field("public", &self.public)
            .field("silu_in_exponents", &self.silu_in_exponents)
            .field("channels", &self.channels)
            .finish()
    }
}

/// One INT8 weight matrix, row-major: its rows are its output channels, so `Embed` holds the
/// transposed embedding, `d_model × vocab` (03 §2, RFC-0012).
#[derive(Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub struct WeightTensor {
    /// The weight type.
    pub ty: WeightType,
    /// The layer, 0 for `Embed` and `LmHead`.
    pub layer: u32,
    /// The rows: output channels.
    pub rows: u32,
    /// The columns: input channels.
    pub cols: u32,
    /// `rows · cols` entries in `[−127, 127]`.
    pub data: Vec<i8>,
}

/// Shows the shape, never a weight (05 §3).
impl fmt::Debug for WeightTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightTensor")
            .field("ty", &self.ty)
            .field("layer", &self.layer)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish_non_exhaustive()
    }
}

/// The calibration data's identity (RFC-0005 'Calibration').
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct CalibrationIdentity {
    /// The dataset.
    pub dataset: Ascii,
    /// Its configuration.
    pub config: Ascii,
    /// Its split.
    pub split: Ascii,
    /// Its commit, 40 hex digits.
    pub revision: Ascii,
    /// The files read, in order.
    pub files: Vec<Ascii>,
    /// The seed.
    pub seed: u64,
    /// The calibration sequences.
    pub sequences: u32,
    /// The held-out sequences.
    pub held_out: u32,
    /// Tokens per sequence.
    pub tokens_per_sequence: u32,
    /// The selection digest.
    pub selection: Digest,
}

/// Where `D` comes from (RFC-0005 'Determinism').
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct Provenance {
    /// The version of `tessera-quant` that wrote `D`.
    pub tessera_quant: Ascii,
    /// Each safetensors file's name and BLAKE3 digest, sorted by name.
    pub checkpoint: Vec<(Ascii, Digest)>,
    /// The calibration data.
    pub calibration: CalibrationIdentity,
    /// The statistics digest of RFC-0005 'Calibration'.
    pub statistics: Digest,
    /// The migration strength chosen.
    pub alpha: Rational,
    /// Library versions, by name.
    pub libraries: Map<Ascii, Ascii>,
}

/// The integer model `D`, artifact `0x0101` (03 §2).
#[derive(Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub struct IntegerModel {
    /// The architecture.
    pub arch: ArchSpec,
    /// The dataflow.
    pub profile: Profile,
    /// The quantization constants.
    pub quant: QuantSpec,
    /// The weights, ordered by `(ty, layer)`.
    pub weights: Vec<WeightTensor>,
    /// Where it comes from.
    pub provenance: Provenance,
}

/// Shows everything but the weights' and channels' entries (05 §3).
impl fmt::Debug for IntegerModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntegerModel")
            .field("arch", &self.arch)
            .field("profile", &self.profile)
            .field("quant", &self.quant)
            .field("weights", &self.weights)
            .field("provenance", &self.provenance)
            .finish()
    }
}
