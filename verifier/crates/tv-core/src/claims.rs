// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The claim list (RFC-0007 'Schema', with RFC-0025's affine sites, RFC-0026's commitment
//! layout, RFC-0027's logits exponent and RFC-0042's SiLU exponents) and its canonical encoding
//! (RFC-0003): decoded from a sealed `.tcl` envelope, schema `0x0201`, version 1 (03 §5), under
//! the caller's decode limits, and encoded back for
//! `claim_list_digest = H("claim-list", TCE(ClaimList))`.
//!
//! Every struct is its fields in declaration order, every enum a `u16` tag, its variant's
//! declaration index, then the variant's fields (RFC-0003 'Composite types'):
//!
//! | Type | Fields, in order |
//! |---|---|
//! | [`ClaimList`] | `protocol_version: u16`, `arch: ArchSpec`, `profile: Profile`, `quant: PublicQuantSpec`, `silu_in_exponents: Vec<(u32, i8)>`, `tables: Vec<TableSpec>`, `gadget_policy: GadgetPolicy`, `layout: CommitmentLayout`, `params: ParamSetId` |
//! | [`ArchSpec`] | `family: ArchFamily`, then `n_layers`, `d_model`, `n_heads`, `n_kv_heads`, `d_head`, `d_ff`, `vocab`, `max_context`, each `u32`, then `rope: RopeSpec`, `rms_eps: Rational` |
//! | [`ArchFamily`] | `LlamaV3` 0 |
//! | [`RopeSpec`] | `theta: u32`, `factor: u16`, `low_freq_factor: u16`, `high_freq_factor: u16`, `original_max: u32`, `bits: u8` |
//! | [`Profile`] | `Servable` 0, `Minimal` 1 |
//! | [`PublicQuantSpec`] | `act_exponents: Vec<(QuantPoint, i8)>`, `residual_exponent: i8`, `residual_bits: u8`, `rsqrt_kappa: u8`, `rms_eps_int: u64`, `exp_sigma: u8`, `score_scales: Vec<(u32, Rational)>`, `shift_class_sets: Vec<(QuantPoint, Vec<i8>)>`, `affine_sites: Vec<(QuantPoint, i8)>`, `rope: RopeSpec`, `logits_bits: u8`, `logits_exponent: i8` |
//! | [`QuantPoint`] | `Embed` 0, `Layer { layer: u32, site: Site }` 1, `FinalNorm` 2 |
//! | [`Site`] | `AttnNorm` 0, `RopeQ` 1, `RopeK` 2, `V` 3, `AttnOut` 4, `AttnResidual` 5, `MlpNorm` 6, `Silu` 7, `Up` 8, `Product` 9, `MlpResidual` 10 |
//! | [`GadgetPolicy`] | `max_log2_rows: u8`, `query_block: u32`, `limb_bits: u8` |
//! | [`CommitmentLayout`] | `advice: AdviceLayoutParams`, `weights: WeightLayoutParams` |
//! | [`AdviceLayoutParams`] | `log2_k: u8`, `log2_inv_rate: u8`, `mask_coeffs: u32`, `row_block: u32`, `segments: u32` |
//! | [`WeightLayoutParams`] | `log2_k: u8`, `log2_inv_rate: u8`, `mask_coeffs: u32`, `blinding_rows: u32` |
//! | [`ParamSetId`] | its one field, `Ascii` (RFC-0014) |
//!
//! `TableSpec` is `tv-tables`' ([`TableSpec::decode`]).
//!
//! # Decoding
//!
//! [`ClaimList::open`] checks, in this order, and the first failure is the error, with its
//! `ERR-CODEC-*` code and offset from the start of the envelope:
//!
//! 1. the envelope, by [`Envelope::open`] in RFC-0003's order: magic, envelope version,
//!    schema id `0x0201` (`ERR-CODEC-MAGIC`), schema version 1 only, since no writer emits 0
//!    (`ERR-CODEC-VERSION`), the reserved field, the size against `max_total_bytes`, the bytes
//!    present, then the digest, BLAKE3 `derive_key("tessera/v1/envelope")` over the header and
//!    payload (`ERR-CODEC-DIGEST`), before any payload byte is decoded;
//! 2. the payload, front to back, with `tv-codec`'s cursor and its errors: every fixed-width
//!    value read whole, every sequence's length checked against `max_vec_len` and then against
//!    the bytes left at its element's minimum size before any element is read, every enum tag
//!    against its type's variants above (`ERR-CODEC-NONCANONICAL` at the tag), every rational
//!    in lowest terms with a positive denominator, the name's bytes printable ASCII;
//! 3. no byte left after the claim list (INV-TCE-4).
//!
//! Depth: the claim list is level 1, and each struct, enum, tuple, sequence, `Rational` and
//! `Ascii` inside it adds one, as RFC-0003 'Limits' counts. A newtype of one field encodes as
//! its field and adds no level of its own, as `tv-tables` reads `TableId(u32)`: so `params`, an
//! `Ascii`, is level 2. The deepest values of a claim list are the sites of its points, level 6.
//!
//! Decoding checks no rule of RFC-0007 'Verifier validation': a claim list that decodes is only
//! well formed. It is the verifier's own input, so one that does not decode is
//! `Inconclusive(ERR-CODEC-*)`, never a rejection (RFC-0043 'Malformed').
//!
//! # Encoding
//!
//! [`ClaimList::to_tce`] and [`ClaimList::digest`] write the same bytes, the second straight into
//! the hash, so the digest allocates nothing but each table's encoding in turn. A claim list that
//! decoded encodes back to its payload byte for byte (INV-DM-1).

use core::fmt;

use tv_codec::{Cursor, DecodeError, DecodeLimits, Envelope, ErrorCode, Expected, Rational, size};
use tv_hash::{Digest, Domain, DomainHasher, hash};
use tv_tables::TableSpec;

/// The envelope a claim list travels in: schema `0x0201`, version 1 (03 §5). A reader of
/// version 1 accepts version 1 only (RFC-0003 'Envelope').
pub const SCHEMA: Expected = Expected {
    schema_id: 0x0201,
    version: 1,
};

/// The public, canonical description of what a proof certifies about `D` (RFC-0007, RFC-0026,
/// RFC-0042). Every field is the prover's word until [`validate`](crate::validation::validate)
/// has checked it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ClaimList {
    /// The protocol's number, `tessera-protocol/{protocol_version}`: 0 until
    /// `tessera-protocol/1` is assigned (09 §3).
    pub protocol_version: u16,
    /// The architecture (03 §2).
    pub arch: ArchSpec,
    /// The profile (RFC-0005 'Profiles').
    pub profile: Profile,
    /// The public quantization data (RFC-0007, RFC-0025, RFC-0027).
    pub quant: PublicQuantSpec,
    /// Each layer's `Silu` input exponent, sorted by layer (RFC-0042).
    pub silu_in_exponents: Vec<(u32, i8)>,
    /// The table set, sorted by `TableId` (RFC-0029).
    pub tables: Vec<TableSpec>,
    /// The gadget policy (RFC-0006 'Gadget selection').
    pub gadget_policy: GadgetPolicy,
    /// The commitment layout, the parameter set's values (RFC-0026).
    pub layout: CommitmentLayout,
    /// The parameter set (RFC-0014).
    pub params: ParamSetId,
}

/// A model's architecture (03 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArchSpec {
    /// The family, Llama 3 only in v1.
    pub family: ArchFamily,
    /// The number of layers.
    pub n_layers: u32,
    /// The residual width.
    pub d_model: u32,
    /// The query heads.
    pub n_heads: u32,
    /// The key and value heads.
    pub n_kv_heads: u32,
    /// A head's width.
    pub d_head: u32,
    /// The MLP's width.
    pub d_ff: u32,
    /// The vocabulary's size.
    pub vocab: u32,
    /// The longest sequence.
    pub max_context: u32,
    /// RoPE's parameters.
    pub rope: RopeSpec,
    /// RMSNorm's epsilon, an exact rational.
    pub rms_eps: Rational,
}

/// An architecture family (03 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArchFamily {
    /// Llama 3, tag 0.
    LlamaV3,
}

/// Llama 3.1's RoPE: the integer parameters of RFC-0008's `Rope` table (03 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RopeSpec {
    /// The base `θ`.
    pub theta: u32,
    /// Llama 3's `factor`.
    pub factor: u16,
    /// Llama 3's `low_freq_factor`.
    pub low_freq_factor: u16,
    /// Llama 3's `high_freq_factor`.
    pub high_freq_factor: u16,
    /// `original_max_position_embeddings`.
    pub original_max: u32,
    /// The constants' fractional bits.
    pub bits: u8,
}

/// A profile (RFC-0005 'Profiles').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// Tag 0: INT8 operands and INT32 accumulators at every weight matmul.
    Servable,
    /// Tag 1: the commitment-minimal dataflow.
    Minimal,
}

/// The public part of `D`'s quantization (RFC-0007 'Schema', RFC-0025, RFC-0027).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PublicQuantSpec {
    /// The exponent `e` of each INT8 tensor's scale `2^e`, sorted by point.
    pub act_exponents: Vec<(QuantPoint, i8)>,
    /// The residual's exponent.
    pub residual_exponent: i8,
    /// The residual's width, 20 in `D`.
    pub residual_bits: u8,
    /// The norms' precision `κ` (RFC-0023).
    pub rsqrt_kappa: u8,
    /// RMSNorm's epsilon as the integer `e` of RFC-0005.
    pub rms_eps_int: u64,
    /// `σ`, the exponentials' fractional bits.
    pub exp_sigma: u8,
    /// Each layer's score scale `σ_q σ_k / √d_head`, sorted by layer.
    pub score_scales: Vec<(u32, Rational)>,
    /// Each `pot` site's class shifts, sorted by point.
    pub shift_class_sets: Vec<(QuantPoint, Vec<i8>)>,
    /// Each affine site's one shift, sorted by point (RFC-0025).
    pub affine_sites: Vec<(QuantPoint, i8)>,
    /// RoPE's parameters, which must be the architecture's.
    pub rope: RopeSpec,
    /// The logits' width.
    pub logits_bits: u8,
    /// A logit `ℓ` stands for `ℓ · 2^logits_exponent` (RFC-0027).
    pub logits_exponent: i8,
}

/// A quantized or rescaled tensor (03 §2), ordered as declared: the embedding, then each layer's
/// points by layer and then by site, then the final norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantPoint {
    /// Tag 0: the INT8 embedding rows.
    Embed,
    /// Tag 1: a site of a layer.
    Layer {
        /// The layer.
        layer: u32,
        /// The site.
        site: Site,
    },
    /// Tag 2: the final norm's output, the LM head's input.
    FinalNorm,
}

/// The rescales of a layer, in the order of RFC-0005 'Integer semantics' (03 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Site {
    /// The attention norm's output.
    AttnNorm,
    /// `q` after RoPE; its classes follow `W_Q`'s channels.
    RopeQ,
    /// `k` after RoPE; its classes follow `W_K`'s channels.
    RopeK,
    /// `v`; its classes follow `W_V`'s.
    V,
    /// The attention output, `servable` only.
    AttnOut,
    /// The O projection's accumulator into the residual; its classes follow `W_O`'s.
    AttnResidual,
    /// The MLP norm's output.
    MlpNorm,
    /// SiLU's output; its classes follow `W_Gate`'s.
    Silu,
    /// The up projection's output, `servable` only; its classes follow `W_Up`'s.
    Up,
    /// The product `a · u8`, `servable` only.
    Product,
    /// The down projection's accumulator into the residual; its classes follow `W_Down`'s.
    MlpResidual,
}

/// A weight matrix's type (03 §2), in declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WeightType {
    /// The transposed embedding, `d_model × vocab`.
    Embed,
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The output projection.
    O,
    /// The MLP's gate.
    Gate,
    /// The MLP's up projection.
    Up,
    /// The MLP's down projection.
    Down,
    /// The LM head.
    LmHead,
}

/// The gadget policy (RFC-0007 'Schema', RFC-0006 'Gadget selection').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GadgetPolicy {
    /// `B_max`, the widest table's `log2` rows, 32 by default.
    pub max_log2_rows: u8,
    /// The attention tiles' query block, 512 by default (RFC-0041).
    pub query_block: u32,
    /// A limb's width, 16 by default.
    pub limb_bits: u8,
}

/// The commitment layout: the parameter set's values (RFC-0026).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CommitmentLayout {
    /// The advice commitment's (RFC-0011).
    pub advice: AdviceLayoutParams,
    /// The weight commitment's (RFC-0012).
    pub weights: WeightLayoutParams,
}

/// The advice commitment's parameters (RFC-0011, RFC-0026).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdviceLayoutParams {
    /// `log2 k`, the message width.
    pub log2_k: u8,
    /// `log2` of the inverse rate.
    pub log2_inv_rate: u8,
    /// `P_A`, each row's mask coefficients.
    pub mask_coeffs: u32,
    /// `B`, the row block.
    pub row_block: u32,
    /// `S`, the segments.
    pub segments: u32,
}

/// The weight commitment's parameters (RFC-0012, RFC-0026).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightLayoutParams {
    /// `log2 k_W`, the largest message width.
    pub log2_k: u8,
    /// `log2` of the inverse rate.
    pub log2_inv_rate: u8,
    /// `P_W`, each row's mask coefficients.
    pub mask_coeffs: u32,
    /// `B_W`, each matrix's blinding rows.
    pub blinding_rows: u32,
}

/// A parameter set's name and version, such as `live-64/1` (RFC-0014 'Parameter sets'):
/// printable ASCII.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParamSetId(String);

// ----- Site facts ------------------------------------------------------------------------------

impl Site {
    /// Every site, in declaration order, which is the order of their tags.
    pub const ALL: [Site; 11] = [
        Site::AttnNorm,
        Site::RopeQ,
        Site::RopeK,
        Site::V,
        Site::AttnOut,
        Site::AttnResidual,
        Site::MlpNorm,
        Site::Silu,
        Site::Up,
        Site::Product,
        Site::MlpResidual,
    ];

    /// The site's TCE tag, its declaration index.
    pub const fn tag(self) -> u16 {
        match self {
            Site::AttnNorm => 0,
            Site::RopeQ => 1,
            Site::RopeK => 2,
            Site::V => 3,
            Site::AttnOut => 4,
            Site::AttnResidual => 5,
            Site::MlpNorm => 6,
            Site::Silu => 7,
            Site::Up => 8,
            Site::Product => 9,
            Site::MlpResidual => 10,
        }
    }

    /// The site of tag `tag`.
    const fn of_tag(tag: u16) -> Option<Site> {
        match tag {
            0 => Some(Site::AttnNorm),
            1 => Some(Site::RopeQ),
            2 => Some(Site::RopeK),
            3 => Some(Site::V),
            4 => Some(Site::AttnOut),
            5 => Some(Site::AttnResidual),
            6 => Some(Site::MlpNorm),
            7 => Some(Site::Silu),
            8 => Some(Site::Up),
            9 => Some(Site::Product),
            10 => Some(Site::MlpResidual),
            _ => None,
        }
    }

    /// Whether `profile`'s template has the site: `AttnOut`, `Up` and `Product` are `servable`'s
    /// alone (03 §2, RFC-0044 'A layer').
    pub(crate) const fn in_profile(self, profile: Profile) -> bool {
        match profile {
            Profile::Servable => true,
            Profile::Minimal => !matches!(self, Site::AttnOut | Site::Up | Site::Product),
        }
    }

    /// Whether the site's output is INT8: every site but the two residual sites, whose output
    /// is the residual (RFC-0005 'Integer semantics').
    pub(crate) const fn is_int8(self) -> bool {
        !self.is_residual()
    }

    /// Whether the site rescales an accumulator into the residual.
    pub(crate) const fn is_residual(self) -> bool {
        matches!(self, Site::AttnResidual | Site::MlpResidual)
    }

    /// "The weight before `p`" (RFC-0044 'Notation'): Q at `RopeQ`, K at `RopeK`, V at `V`, O at
    /// `AttnResidual`, Gate at `Silu`, Up at `Up`, Down at `MlpResidual`; none at the norms,
    /// `AttnOut` and `Product`.
    pub(crate) const fn weight_before(self) -> Option<WeightType> {
        match self {
            Site::RopeQ => Some(WeightType::Q),
            Site::RopeK => Some(WeightType::K),
            Site::V => Some(WeightType::V),
            Site::AttnResidual => Some(WeightType::O),
            Site::Silu => Some(WeightType::Gate),
            Site::Up => Some(WeightType::Up),
            Site::MlpResidual => Some(WeightType::Down),
            Site::AttnNorm | Site::AttnOut | Site::MlpNorm | Site::Product => None,
        }
    }

    /// Whether the site may be affine: it follows the Q, K, V, O, up or down projection
    /// (RFC-0025 'Public data'). `Silu` follows the gate and stays `pot`.
    pub(crate) const fn may_be_affine(self) -> bool {
        matches!(
            self.weight_before(),
            Some(
                WeightType::Q
                    | WeightType::K
                    | WeightType::V
                    | WeightType::O
                    | WeightType::Up
                    | WeightType::Down
            )
        )
    }
}

// ----- ParamSetId ------------------------------------------------------------------------------

impl ParamSetId {
    /// The identifier `name`, or `None` when a byte of it is outside printable ASCII,
    /// `0x20..=0x7E`, which TCE's `Ascii` refuses.
    pub fn new(name: &str) -> Option<Self> {
        name.bytes()
            .all(|b| (0x20..=0x7E).contains(&b))
            .then(|| Self(name.to_owned()))
    }

    /// The name, such as `ni-100/1`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A copy whose memory is reserved fallibly: `None` when it cannot be. A claim list's name
    /// is as long as its decode limits allow.
    pub(crate) fn try_clone(&self) -> Option<Self> {
        let mut copy = String::new();
        copy.try_reserve_exact(self.0.len()).ok()?;
        copy.push_str(&self.0);
        Some(Self(copy))
    }

    /// Decodes the identifier, an `Ascii` copied out of the input into memory reserved
    /// fallibly: memory that cannot be reserved is `ERR-RES-MEMORY` at the name.
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        let at = c.offset();
        let text = c.ascii()?;
        let mut copy = String::new();
        copy.try_reserve_exact(text.len())
            .map_err(|_| DecodeError::new(ErrorCode::Memory, at))?;
        copy.push_str(text);
        Ok(Self(copy))
    }
}

impl fmt::Display for ParamSetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ----- Decoding --------------------------------------------------------------------------------

/// `min_size(QuantPoint)`: the tag of `Embed`, which has no field.
const POINT: u64 = size::TAG;

/// `min_size((QuantPoint, i8))`.
const POINT_EXPONENT: u64 = size::sum(&[POINT, size::I8]);

/// `min_size((QuantPoint, Vec<i8>))`.
const POINT_SHIFTS: u64 = size::sum(&[POINT, size::SEQUENCE]);

/// `min_size((u32, Rational))`.
const LAYER_SCALE: u64 = size::sum(&[size::U32, size::RATIONAL]);

/// `min_size((u32, i8))`.
const LAYER_EXPONENT: u64 = size::sum(&[size::U32, size::I8]);

impl ClaimList {
    /// Opens a sealed claim list, `claims.tcl` (RFC-0003 'Envelope', 03 §5), and decodes it,
    /// in the order of the module documentation. Its digest function is `tv-hash`'s.
    pub fn open(input: &[u8], limits: &DecodeLimits) -> Result<Self, DecodeError> {
        let envelope = Envelope::open(input, SCHEMA, limits, |signed| {
            hash(Domain::Envelope, signed)
        })?;
        envelope.decode_payload(|c, _version| Self::decode(c))
    }

    /// Decodes `TCE(ClaimList)` at the cursor, one level deeper.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let protocol_version = c.u16()?;
            let arch = ArchSpec::decode(c)?;
            let profile = Profile::decode(c)?;
            let quant = PublicQuantSpec::decode(c)?;
            let silu_in_exponents =
                c.vec(LAYER_EXPONENT, |c| c.nested(|c| Ok((c.u32()?, c.i8()?))))?;
            let tables = c.vec(TableSpec::MIN_SIZE, TableSpec::decode)?;
            let gadget_policy = GadgetPolicy::decode(c)?;
            let layout = CommitmentLayout::decode(c)?;
            let params = ParamSetId::decode(c)?;
            Ok(Self {
                protocol_version,
                arch,
                profile,
                quant,
                silu_in_exponents,
                tables,
                gadget_policy,
                layout,
                params,
            })
        })
    }

    /// `TCE(ClaimList)`: the fields in declaration order (RFC-0003 'Composite types').
    pub fn to_tce(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    /// `claim_list_digest = H("claim-list", TCE(ClaimList))` (RFC-0007, RFC-0004), the encoding
    /// written straight into the hash.
    pub fn digest(&self) -> Digest {
        let mut hasher = DomainHasher::new(Domain::ClaimList);
        self.encode(&mut hasher);
        hasher.finalize()
    }

    fn encode(&self, out: &mut impl Sink) {
        out.put(&self.protocol_version.to_le_bytes());
        self.arch.encode(out);
        out.put(&self.profile.tag().to_le_bytes());
        self.quant.encode(out);
        put_length(out, self.silu_in_exponents.len());
        for (layer, exponent) in &self.silu_in_exponents {
            out.put(&layer.to_le_bytes());
            out.put(&exponent.to_le_bytes());
        }
        put_length(out, self.tables.len());
        for table in &self.tables {
            out.put(&table.to_tce());
        }
        self.gadget_policy.encode(out);
        self.layout.encode(out);
        put_length(out, self.params.0.len());
        out.put(self.params.0.as_bytes());
    }
}

impl ArchSpec {
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let family = ArchFamily::decode(c)?;
            let n_layers = c.u32()?;
            let d_model = c.u32()?;
            let n_heads = c.u32()?;
            let n_kv_heads = c.u32()?;
            let d_head = c.u32()?;
            let d_ff = c.u32()?;
            let vocab = c.u32()?;
            let max_context = c.u32()?;
            let rope = RopeSpec::decode(c)?;
            let rms_eps = c.rational()?;
            Ok(Self {
                family,
                n_layers,
                d_model,
                n_heads,
                n_kv_heads,
                d_head,
                d_ff,
                vocab,
                max_context,
                rope,
                rms_eps,
            })
        })
    }

    fn encode(&self, out: &mut impl Sink) {
        out.put(&self.family.tag().to_le_bytes());
        for value in [
            self.n_layers,
            self.d_model,
            self.n_heads,
            self.n_kv_heads,
            self.d_head,
            self.d_ff,
            self.vocab,
            self.max_context,
        ] {
            out.put(&value.to_le_bytes());
        }
        self.rope.encode(out);
        put_rational(out, self.rms_eps);
    }
}

impl ArchFamily {
    /// The variant's TCE tag.
    const fn tag(self) -> u16 {
        match self {
            ArchFamily::LlamaV3 => 0,
        }
    }

    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.enumeration(1, |_, _| Ok(ArchFamily::LlamaV3))
    }
}

impl RopeSpec {
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let theta = c.u32()?;
            let factor = c.u16()?;
            let low_freq_factor = c.u16()?;
            let high_freq_factor = c.u16()?;
            let original_max = c.u32()?;
            let bits = c.u8()?;
            Ok(Self {
                theta,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max,
                bits,
            })
        })
    }

    fn encode(&self, out: &mut impl Sink) {
        out.put(&self.theta.to_le_bytes());
        out.put(&self.factor.to_le_bytes());
        out.put(&self.low_freq_factor.to_le_bytes());
        out.put(&self.high_freq_factor.to_le_bytes());
        out.put(&self.original_max.to_le_bytes());
        out.put(&[self.bits]);
    }
}

impl Profile {
    /// The variant's TCE tag.
    const fn tag(self) -> u16 {
        match self {
            Profile::Servable => 0,
            Profile::Minimal => 1,
        }
    }

    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.enumeration(2, |_, tag| {
            Ok(if tag == 0 {
                Profile::Servable
            } else {
                Profile::Minimal
            })
        })
    }
}

impl PublicQuantSpec {
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let act_exponents = c.vec(POINT_EXPONENT, point_exponent)?;
            let residual_exponent = c.i8()?;
            let residual_bits = c.u8()?;
            let rsqrt_kappa = c.u8()?;
            let rms_eps_int = c.u64()?;
            let exp_sigma = c.u8()?;
            let score_scales =
                c.vec(LAYER_SCALE, |c| c.nested(|c| Ok((c.u32()?, c.rational()?))))?;
            let shift_class_sets = c.vec(POINT_SHIFTS, |c| {
                c.nested(|c| Ok((QuantPoint::decode(c)?, c.vec(size::I8, Cursor::i8)?)))
            })?;
            let affine_sites = c.vec(POINT_EXPONENT, point_exponent)?;
            let rope = RopeSpec::decode(c)?;
            let logits_bits = c.u8()?;
            let logits_exponent = c.i8()?;
            Ok(Self {
                act_exponents,
                residual_exponent,
                residual_bits,
                rsqrt_kappa,
                rms_eps_int,
                exp_sigma,
                score_scales,
                shift_class_sets,
                affine_sites,
                rope,
                logits_bits,
                logits_exponent,
            })
        })
    }

    fn encode(&self, out: &mut impl Sink) {
        put_point_exponents(out, &self.act_exponents);
        out.put(&self.residual_exponent.to_le_bytes());
        out.put(&[self.residual_bits, self.rsqrt_kappa]);
        out.put(&self.rms_eps_int.to_le_bytes());
        out.put(&[self.exp_sigma]);
        put_length(out, self.score_scales.len());
        for (layer, scale) in &self.score_scales {
            out.put(&layer.to_le_bytes());
            put_rational(out, *scale);
        }
        put_length(out, self.shift_class_sets.len());
        for (point, shifts) in &self.shift_class_sets {
            point.encode(out);
            put_length(out, shifts.len());
            for shift in shifts {
                out.put(&shift.to_le_bytes());
            }
        }
        put_point_exponents(out, &self.affine_sites);
        self.rope.encode(out);
        out.put(&[self.logits_bits]);
        out.put(&self.logits_exponent.to_le_bytes());
    }
}

/// `(QuantPoint, i8)`, a tuple one level deeper.
fn point_exponent(c: &mut Cursor<'_>) -> Result<(QuantPoint, i8), DecodeError> {
    c.nested(|c| Ok((QuantPoint::decode(c)?, c.i8()?)))
}

impl QuantPoint {
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        let at = c.offset();
        c.enumeration(3, |c, tag| match tag {
            0 => Ok(QuantPoint::Embed),
            1 => {
                let layer = c.u32()?;
                let site = decode_site(c)?;
                Ok(QuantPoint::Layer { layer, site })
            }
            2 => Ok(QuantPoint::FinalNorm),
            // `enumeration` has refused every tag of 3 or more.
            _ => Err(DecodeError::new(ErrorCode::Internal, at)),
        })
    }

    fn encode(&self, out: &mut impl Sink) {
        match self {
            QuantPoint::Embed => out.put(&0u16.to_le_bytes()),
            QuantPoint::Layer { layer, site } => {
                out.put(&1u16.to_le_bytes());
                out.put(&layer.to_le_bytes());
                out.put(&site.tag().to_le_bytes());
            }
            QuantPoint::FinalNorm => out.put(&2u16.to_le_bytes()),
        }
    }
}

/// A [`Site`]: an enum of eleven variants without fields.
fn decode_site(c: &mut Cursor<'_>) -> Result<Site, DecodeError> {
    let at = c.offset();
    c.enumeration(11, |_, tag| {
        // `enumeration` has refused every tag of 11 or more.
        Site::of_tag(tag).ok_or(DecodeError::new(ErrorCode::Internal, at))
    })
}

impl GadgetPolicy {
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let max_log2_rows = c.u8()?;
            let query_block = c.u32()?;
            let limb_bits = c.u8()?;
            Ok(Self {
                max_log2_rows,
                query_block,
                limb_bits,
            })
        })
    }

    fn encode(&self, out: &mut impl Sink) {
        out.put(&[self.max_log2_rows]);
        out.put(&self.query_block.to_le_bytes());
        out.put(&[self.limb_bits]);
    }
}

impl CommitmentLayout {
    fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let advice = c.nested(|c| {
                let log2_k = c.u8()?;
                let log2_inv_rate = c.u8()?;
                let mask_coeffs = c.u32()?;
                let row_block = c.u32()?;
                let segments = c.u32()?;
                Ok(AdviceLayoutParams {
                    log2_k,
                    log2_inv_rate,
                    mask_coeffs,
                    row_block,
                    segments,
                })
            })?;
            let weights = c.nested(|c| {
                let log2_k = c.u8()?;
                let log2_inv_rate = c.u8()?;
                let mask_coeffs = c.u32()?;
                let blinding_rows = c.u32()?;
                Ok(WeightLayoutParams {
                    log2_k,
                    log2_inv_rate,
                    mask_coeffs,
                    blinding_rows,
                })
            })?;
            Ok(Self { advice, weights })
        })
    }

    fn encode(&self, out: &mut impl Sink) {
        let advice = &self.advice;
        out.put(&[advice.log2_k, advice.log2_inv_rate]);
        out.put(&advice.mask_coeffs.to_le_bytes());
        out.put(&advice.row_block.to_le_bytes());
        out.put(&advice.segments.to_le_bytes());
        let weights = &self.weights;
        out.put(&[weights.log2_k, weights.log2_inv_rate]);
        out.put(&weights.mask_coeffs.to_le_bytes());
        out.put(&weights.blinding_rows.to_le_bytes());
    }
}

// ----- Encoding --------------------------------------------------------------------------------

/// Where TCE bytes go: a buffer, or the claim list's hash.
trait Sink {
    fn put(&mut self, bytes: &[u8]);
}

impl Sink for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

impl Sink for DomainHasher {
    fn put(&mut self, bytes: &[u8]) {
        self.update(bytes);
    }
}

/// A sequence's `u64` length. Every supported target has pointers of at most 64 bits, so the
/// conversion is exact.
fn put_length(out: &mut impl Sink, len: usize) {
    out.put(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

/// A `Rational`: its `i64` numerator, then its `u64` denominator.
fn put_rational(out: &mut impl Sink, value: Rational) {
    out.put(&value.num().to_le_bytes());
    out.put(&value.den().to_le_bytes());
}

/// A `Vec<(QuantPoint, i8)>`.
fn put_point_exponents(out: &mut impl Sink, entries: &[(QuantPoint, i8)]) {
    put_length(out, entries.len());
    for (point, exponent) in entries {
        point.encode(out);
        out.put(&exponent.to_le_bytes());
    }
}
