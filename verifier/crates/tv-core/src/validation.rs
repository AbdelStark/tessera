// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claim-list validation (RFC-0007 'Verifier validation', RFC-0019 'Check order' steps 2 and 3),
//! in the order RFC-0039 'The checks' fixes, each check with the rejection RFC-0039 and RFC-0043
//! name for it. The first check that fails is the verdict.
//!
//! # The checks
//!
//! | Step | Check | Failure |
//! |---|---|---|
//! | RFC-0019 step 2 | `params` names a set of [`ParamSet::accept`]: `live-64/1`, `ni-100/1`, and `test-small/1` with the `test-params` feature only | `Params { name }`, the claim list's name |
//! | 1 | `protocol_version` is [`PROTOCOL_VERSION`] | `ProtocolVersion` |
//! | 2.1 | the architecture, below | `Architecture` |
//! | 2.2 | the public quantization data, below | `Quantization` |
//! | 2.3 | every layer's sites rescaled alike (RFC-0038) | `Rescaling { site }`, the first site in 03 §2's declaration order that breaks it |
//! | 3 | S1 to S3 of RFC-0005 on the derived graph, below | `Profile` |
//! | 4 | each table regenerated from its definition ([`Table::from_spec`]), in `TableId` order | `Table { table }`, the first in `TableId` order |
//! | 5.1 | `1 ≤ max_log2_rows ≤ 32` | `GadgetPolicy` |
//! | 5.2 | independently derived integer ranges, gadget domains, field bounds and feasible tables | `Gadget { gadget }`, first in node order |
//! | 5.3 | the complete derived table sequence equals the claim list, id for id | `Table { table }`, first differing id |
//! | 6 | the commitment layout is the parameter set's ([`ParamSet::layout`]), and `query_block` a power of two (RFC-0041 'Production order') | `Layout` |
//! | 7 | the scoring and anchor templates name supported variants | `Template`, once they join the claim list (RFC-0026): today there is none to check |
//!
//! Every failure but the parameter set's is `Reject(StatementMismatch { field })`, and the
//! parameter set's is `Reject(Params { name })`. Within one check the rules give one field, so
//! their order does not matter. What validation does not do yet:
//!
//! - **Step 6's advice layout** (#106): that the derived advice layout fits within the protocol's
//!   maximum of `2^32` rows at `max_context` (RFC-0047) needs the complete model-to-tile
//!   derivation, including the remaining gadgets' committed columns.
//! - **`ConstraintSystemDigest`** (RFC-0007), which #120 defines.
//!
//! ## Step 2.1: the architecture
//!
//! `n_layers ≥ 1`; `n_kv_heads ≥ 1` divides `n_heads ≥ 1`, with `n_heads / n_kv_heads` a power
//! of two (RFC-0047); `d_head` even, from 2 to `2^16 − 2`;
//! `d_model = n_heads · d_head`; `d_ff ≥ 1`; `1 ≤ vocab ≤ 2^18`; `1 ≤ max_context ≤ 2^17`. The
//! family is Llama 3 by decoding: `ArchFamily` has no other variant.
//!
//! ## Step 2.2: the public quantization data
//!
//! The **rescale points** are each layer's sites of the profile, every site in `servable` and all
//! but `AttnOut`, `Up` and `Product` in `minimal`, then `FinalNorm`. The **INT8 points** are the
//! embedding, each layer's sites of the profile but the two residual sites, and the final norm.
//! Both are in 03 §2's order, layer by layer and site by site in declaration order. The rules:
//!
//! - `quant.rope` is `arch.rope`;
//! - `act_exponents` lists exactly the INT8 points, in order, once each;
//! - `shift_class_sets` and `affine_sites` list exactly the rescale points between them, each
//!   point in one list, both in order, once each. Both lists are walked against the rescale
//!   points in order, so a point out of order, repeated, in both lists, in neither or no rescale
//!   point fails the same rule;
//! - an affine site follows the Q, K, V, O, up or down projection (`RopeQ`, `RopeK`, `V`,
//!   `AttnResidual`, `Up`, `MlpResidual`), with a shift from 1 to 63;
//! - a `pot` site has one to four classes, each from 1 to 63, and one class if it follows no
//!   weight (the norms, `AttnOut`, `Product` and the final norm);
//! - `score_scales` holds `(ℓ, s)` for `ℓ = 0, …, n_layers − 1` in order, each `s > 0`;
//! - for each layer's `SoftmaxRow`, the narrowest `key_bits` from the spread's width,
//!   `bitlen(128 · 255 · d_head)`, up whose `Exp { key_bits, exp_sigma, scale }` maps its last key
//!   `2^{key_bits} − 1` to 0 ([`exp_maps_to_zero`]) exists, and RFC-0008 allows that table
//!   ([`TableKind::check_parameters`]): `1 ≤ key_bits ≤ 32`, `σ ≤ 32`, `scale > 0`, its zero tail
//!   starting by key `2^24` (RFC-0008 'Table specifications', RFC-0030 'Range analysis and
//!   validation'). A table widened past 24 bits never qualifies: its last key rounds to 0 before
//!   its zero tail, which then starts past `2^24`. A wide head's spread can still be wider, such
//!   as 26 bits for heads of 2,048, when its own table qualifies;
//! - `SoftmaxRow`'s reciprocal is unique (RFC-0009 Lemma 6): `(2·max_context − 1)·2^{2·exp_sigma}
//!   < P`, computed exactly in 128 bits. That is `exp_sigma ≤ 23` at `max_context = 2^17`, and
//!   `exp_sigma ≤ 31` at any context, inside RFC-0008's `σ ≤ 32`;
//! - `silu_in_exponents` holds `(ℓ, e)` for `ℓ = 0, …, n_layers − 1` in order;
//! - `2 ≤ residual_bits ≤ 31`;
//! - the embedding's shift `s_e = e_Embed − residual_exponent` is non-negative (RFC-0005 'Scale
//!   policy'), so that its constant `2^{s_e}` is an integer, and keeps
//!   `127 · 2^{s_e} < 2^{residual_bits − 1}`;
//! - `rms_eps_int ≥ 1` and `rsqrt_kappa ≤ 31`.
//!
//! RFC-0007 states `n_heads ≥ 1`, `s_e ≥ 0` and the `Exp` rule since #107, and Lemma 6's bound
//! since #376.
//!
//! ## Step 2.3: every layer rescaled alike
//!
//! For each site of the profile, in declaration order: it is affine in every layer or in none, and
//! as a `pot` site it has one class in every layer or several in every layer. The number of
//! classes may differ between layers when it is more than one.
//!
//! ## Step 3: the profile on the derived graph
//!
//! The derived graph is [`lower`]'s for a sequence of `max_context` tokens: node, tensor, table
//! and gadget ids do not depend on the length, and the checks below read no shape that does.
//! Every public part step 2 accepts lowers, since its `Exp` widths exist. S1 to S3 of RFC-0005
//! 'Profiles', as RFC-0006 'Llama lowering' reads them, at every node:
//!
//! - **S1.** A weight matmul's operands are INT8, `[−128, 127]`, except, in `minimal`, the O and
//!   down projections' inputs. The weight is, by its declared range `[−127, 127]`. The input's
//!   range is the one its producer's lookups enforce (RFC-0006 'Range analysis'): `[lo, hi]` after
//!   a `Requant`, `[0, 1]` after a `OneHot`. An input of any other producer is not shown INT8
//!   here, and fails S1 when S1 applies to it.
//! - **S2** (`servable`). A weight matmul's accumulator fits INT32, `[−2^31, 2^31 − 1]`, in the
//!   worst case: the public bound `127 · fan_in · max|x|` of RFC-0006, with `fan_in` the
//!   weight's columns, or the weight's range `[−127, 127]` when the operand is a `OneHot`'s.
//! - **S3.** Every `TableMap` applies a `Silu` table, and no scoring or anchor node is part of
//!   `D`: the lowering creates none.
//!
//! The lowering creates every weight matmul from a rescale or a `OneHot`, but for `minimal`'s O
//! and down projections, so S1 and S3 hold for every claim list that passes step 2: only S2 can
//! fail, in `servable`, when `127 · 128 · max(d_model, d_ff)` passes `2^31 − 1`.
//!
//! # Cost
//!
//! Steps 1 and 2 are linear in the claim list, step 2's `Exp` rule testing at most seventeen
//! keys and one table's parameters a layer, and step 3 is linear in its layers. Step 4 generates
//! each table
//! of the claim list, as `tv-tables` documents: a dense core of at most `2^24` values each.
//! Nothing allocates infallibly but the regenerated tables' cores, which `tv-tables` reserves
//! fallibly too; memory that cannot be reserved is `Inconclusive(ERR-RES-MEMORY)`.

use core::fmt;

use tv_codec::{DecodeError, DecodeLimits, Rational};
use tv_field::Fp;
use tv_tables::{Table, TableError, TableId, TableKind, exp_maps_to_zero};

use crate::claims::{
    ArchSpec, ClaimList, ParamSetId, Profile, PublicQuantSpec, QuantPoint, Site, WeightType,
};
use crate::lowering::{Graph, LowerError, Node, Op, lower, spread_bits};
use crate::params::ParamSet;

/// The protocol version this verifier implements: 0, the number every artifact carries until
/// `tessera-protocol/1` is assigned, when the first end-to-end proof verifies (09 §3, RFC-0036).
pub const PROTOCOL_VERSION: u16 = 0;

/// A gadget's index: gadget-backed nodes numbered in node order (RFC-0006 'Gadget selection').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GadgetId(pub u32);

/// The payload of `Reject(StatementMismatch)` (RFC-0039 'The type'): one variant per check, in
/// check order, each holding the public identifier that locates the failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StatementField {
    /// Step 1: a protocol version the verifier does not implement.
    ProtocolVersion,
    /// Step 2.1: the architecture.
    Architecture,
    /// Step 2.2: the public quantization data.
    Quantization,
    /// Step 2.3: a site not rescaled alike in every layer (RFC-0038).
    Rescaling {
        /// The first site in 03 §2's declaration order that breaks the rule.
        site: Site,
    },
    /// Step 3: the profile's constraints S1 to S3.
    Profile,
    /// Steps 4 and 5.3: a table.
    Table {
        /// The first table in `TableId` order that fails.
        table: TableId,
    },
    /// Step 5.1: `max_log2_rows`.
    GadgetPolicy,
    /// Step 5.2: a gadget-backed node.
    Gadget {
        /// The first gadget in node order that fails.
        gadget: GadgetId,
    },
    /// Step 6: the commitment parameters or the advice layout.
    Layout,
    /// Step 7: the scoring and anchor templates.
    Template,
    /// RFC-0019 step 4: the statement's claim-list digest.
    ClaimListDigest,
    /// RFC-0019 step 4: the statement's weights root.
    WeightsRoot,
    /// RFC-0019 step 4: the statement's weights epoch.
    WeightsEpoch,
    /// RFC-0019 step 4: the statement's manifest digest.
    ManifestDigest,
    /// RFC-0019 step 4: the statement's batch.
    Batch,
    /// RFC-0019 step 4: the statement's predictor.
    Predictor,
    /// RFC-0019 step 4: `Round1`'s `layout_digest` (RFC-0031).
    LayoutDigest,
}

impl StatementField {
    /// The variant's TCE tag, its declaration index (RFC-0039 'The type').
    pub const fn tag(&self) -> u16 {
        match self {
            StatementField::ProtocolVersion => 0,
            StatementField::Architecture => 1,
            StatementField::Quantization => 2,
            StatementField::Rescaling { .. } => 3,
            StatementField::Profile => 4,
            StatementField::Table { .. } => 5,
            StatementField::GadgetPolicy => 6,
            StatementField::Gadget { .. } => 7,
            StatementField::Layout => 8,
            StatementField::Template => 9,
            StatementField::ClaimListDigest => 10,
            StatementField::WeightsRoot => 11,
            StatementField::WeightsEpoch => 12,
            StatementField::ManifestDigest => 13,
            StatementField::Batch => 14,
            StatementField::Predictor => 15,
            StatementField::LayoutDigest => 16,
        }
    }
}

impl fmt::Display for StatementField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatementField::Rescaling { site } => write!(f, "Rescaling {{ site: {site:?} }}"),
            StatementField::Table { table } => write!(f, "Table {{ table: {} }}", table.0),
            StatementField::Gadget { gadget } => write!(f, "Gadget {{ gadget: {} }}", gadget.0),
            other => write!(f, "{other:?}"),
        }
    }
}

/// Why a claim list was not validated, and what verdict that is in 04 §2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaimsError {
    /// The claim list did not decode: `Inconclusive` with the decoder's code, `ERR-CODEC-*` for
    /// a fault of the bytes. A claim list is the verifier's own input, never the prover's, so
    /// no decoding failure is a rejection (RFC-0043 'Malformed', RFC-0019 step 1).
    Decode(DecodeError),
    /// `Reject(Params { name })`: the claim list names a parameter set this verifier does not
    /// accept (RFC-0019 step 2, RFC-0043 'Parameter sets').
    Params {
        /// The name the claim list gives.
        name: ParamSetId,
    },
    /// `Reject(StatementMismatch { field })`: the claim list breaks a rule of RFC-0007
    /// 'Verifier validation' (RFC-0039).
    StatementMismatch {
        /// The check that failed first.
        field: StatementField,
    },
    /// `Inconclusive(ERR-RES-MEMORY)`: memory for the lowering, a table or a copy of the
    /// claim list's name could not be reserved.
    Memory,
    /// `Inconclusive(ERR-INTERNAL)`: a step that the checks before it make infallible failed, a
    /// bug of this crate that no input reaches.
    Internal,
}

impl ClaimsError {
    /// Whether the claim list is at fault: `Reject(reason)` in 04 §2. Otherwise the verifier
    /// could not evaluate it: `Inconclusive(error)`.
    pub const fn is_rejection(&self) -> bool {
        matches!(
            self,
            ClaimsError::Params { .. } | ClaimsError::StatementMismatch { .. }
        )
    }

    /// The identifier of 04 §2 and §3: `REJ-PARAMS`, `REJ-STATEMENT`, the decoder's code, or
    /// `ERR-RES-MEMORY` and `ERR-INTERNAL` for the verifier's faults.
    pub const fn code(&self) -> &'static str {
        match self {
            ClaimsError::Decode(error) => error.code().id(),
            ClaimsError::Params { .. } => "REJ-PARAMS",
            ClaimsError::StatementMismatch { .. } => "REJ-STATEMENT",
            ClaimsError::Memory => "ERR-RES-MEMORY",
            ClaimsError::Internal => "ERR-INTERNAL",
        }
    }
}

impl fmt::Display for ClaimsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClaimsError::Decode(error) => write!(f, "the claim list does not decode: {error}"),
            ClaimsError::Params { name } => {
                write!(f, "the parameter set {name} is not accepted")
            }
            ClaimsError::StatementMismatch { field } => {
                write!(f, "the claim list fails validation: {field}")
            }
            ClaimsError::Memory => f.write_str("memory for claim-list validation ran out"),
            ClaimsError::Internal => f.write_str("claim-list validation failed internally"),
        }
    }
}

impl std::error::Error for ClaimsError {}

/// A validated claim list's derived data: the parameter set it names, its tables regenerated in
/// `TableId` order, the graph its lowering derives for `max_context` tokens, and the integer
/// ranges used to check each gadget and derive the canonical table sequence. Step 6's complete
/// advice layout (#106) remains outstanding: see the module documentation.
#[derive(Debug)]
pub struct Validated {
    params: ParamSet,
    tables: Vec<Table>,
    graph: Graph,
    ranges: Vec<crate::ranges::Interval>,
}

impl Validated {
    /// The parameter set the claim list names.
    pub fn params(&self) -> ParamSet {
        self.params
    }

    /// The claim list's tables, each regenerated from its definition, in `TableId` order.
    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    /// Integer ranges independently derived from public data, indexed by tensor id (SP-7).
    pub fn ranges(&self) -> &[crate::ranges::Interval] {
        &self.ranges
    }

    /// The derived graph, for a sequence of `max_context` tokens.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
}

/// Opens a sealed claim list and validates it: [`ClaimList::open`]'s failure is
/// [`ClaimsError::Decode`], then [`validate`]'s checks.
pub fn load(input: &[u8], limits: &DecodeLimits) -> Result<(ClaimList, Validated), ClaimsError> {
    let claims = ClaimList::open(input, limits).map_err(ClaimsError::Decode)?;
    let validated = validate(&claims)?;
    Ok((claims, validated))
}

/// Validates `claims` in the order of the module documentation; the first check that fails is
/// the error.
pub fn validate(claims: &ClaimList) -> Result<Validated, ClaimsError> {
    // RFC-0019 step 2: the parameter set.
    let Some(params) = ParamSet::accept(&claims.params) else {
        let name = claims.params.try_clone().ok_or(ClaimsError::Memory)?;
        return Err(ClaimsError::Params { name });
    };
    // Step 1: the protocol version.
    if claims.protocol_version != PROTOCOL_VERSION {
        return Err(mismatch(StatementField::ProtocolVersion));
    }
    // Step 2: the architecture and the quantization data.
    let public = check_public(claims)?;
    // Step 3: the profile on the derived graph.
    // Step 2 accepted the public part, so it lowers: only memory can fail.
    let graph = match lower(&public, claims.arch.max_context) {
        Ok(graph) => graph,
        Err(LowerError::Memory) => return Err(ClaimsError::Memory),
        Err(LowerError::Length { .. } | LowerError::Internal) => {
            return Err(ClaimsError::Internal);
        }
    };
    if !profile_holds(&graph, claims.profile)? {
        return Err(mismatch(StatementField::Profile));
    }
    // Step 4: the tables.
    let tables = regenerate(&claims.tables)?;
    // Step 5.1: the gadget policy.
    if !(1..=32).contains(&claims.gadget_policy.max_log2_rows) {
        return Err(mismatch(StatementField::GadgetPolicy));
    }
    let ranges = crate::range_checks::check(&graph, claims.gadget_policy, &claims.tables)?;
    // Step 6: the commitment layout and the query block.
    if claims.layout != params.layout() || !claims.gadget_policy.query_block.is_power_of_two() {
        return Err(mismatch(StatementField::Layout));
    }
    // TODO(#106): step 6's last rule, that the advice layout derived from `graph` fits within the
    // RFC-0047 maximum of 2^32 rows at max_context, needs the complete model-to-tile derivation
    // for RFC-0011, including the remaining gadgets' committed columns.
    //
    // Step 7: `scoring` and `anchor` are not in the claim list (RFC-0026), so no template is
    // named and none can fail. Their check joins with them.
    Ok(Validated {
        params,
        tables,
        graph,
        ranges,
    })
}

/// `Reject(StatementMismatch { field })`.
const fn mismatch(field: StatementField) -> ClaimsError {
    ClaimsError::StatementMismatch { field }
}

// ----- Step 2: the public part ---------------------------------------------------------------

/// How a site rescales in one layer: its `pot` classes, or its affine shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rescale<'a> {
    /// A `pot` site's shifts, one per class.
    Pot(&'a [i8]),
    /// An affine site's one shift.
    Affine(i8),
}

/// One layer's public quantization data, by site.
#[derive(Clone, Copy, Debug)]
struct LayerPart<'a> {
    /// Each site's rescale, by tag; `None` for a site the profile lacks.
    rescales: [Option<Rescale<'a>>; 11],
    /// Each INT8 site's exponent, by tag.
    exponents: [Option<i8>; 11],
    /// The score scale.
    score_scale: Rational,
    /// SiLU's input exponent.
    silu_in_exponent: i8,
}

/// A claim list's public part that steps 2.1 to 2.3 have accepted, indexed by layer: what the
/// lowering reads (RFC-0006 'Llama lowering', RFC-0042). Only [`check_public`] makes one.
#[derive(Clone, Debug)]
pub struct PublicPart<'a> {
    claims: &'a ClaimList,
    layers: Vec<LayerPart<'a>>,
    final_norm: Rescale<'a>,
    embed_shift: u32,
}

impl<'a> PublicPart<'a> {
    pub(crate) fn arch(&self) -> &'a ArchSpec {
        &self.claims.arch
    }

    pub(crate) fn profile(&self) -> Profile {
        self.claims.profile
    }

    pub(crate) fn quant(&self) -> &'a PublicQuantSpec {
        &self.claims.quant
    }

    /// The number of layers.
    pub(crate) fn layers(&self) -> u32 {
        self.claims.arch.n_layers
    }

    fn layer(&self, layer: u32) -> Option<&LayerPart<'a>> {
        self.layers.get(usize::try_from(layer).ok()?)
    }

    /// How `site` rescales in `layer`; `None` for a site the profile lacks.
    pub(crate) fn rescale(&self, layer: u32, site: Site) -> Option<Rescale<'a>> {
        *self.layer(layer)?.rescales.get(usize::from(site.tag()))?
    }

    /// How the final norm rescales: one class.
    pub(crate) fn final_norm(&self) -> Rescale<'a> {
        self.final_norm
    }

    /// An INT8 site's exponent in `layer`.
    pub(crate) fn exponent(&self, layer: u32, site: Site) -> Option<i8> {
        *self.layer(layer)?.exponents.get(usize::from(site.tag()))?
    }

    /// The embedding's shift into the residual, `s_e = e_Embed − residual_exponent`, from 0 to 23.
    pub(crate) fn embed_shift(&self) -> u32 {
        self.embed_shift
    }

    pub(crate) fn score_scale(&self, layer: u32) -> Option<Rational> {
        Some(self.layer(layer)?.score_scale)
    }

    pub(crate) fn silu_in_exponent(&self, layer: u32) -> Option<i8> {
        Some(self.layer(layer)?.silu_in_exponent)
    }
}

/// Steps 2.1 to 2.3 (RFC-0007 'Verifier validation' step 2): the architecture, the public
/// quantization data and the rule that every layer is rescaled alike, in this order. What they
/// accept is the public part the lowering reads.
pub fn check_public(claims: &ClaimList) -> Result<PublicPart<'_>, ClaimsError> {
    if !architecture_holds(&claims.arch) {
        return Err(mismatch(StatementField::Architecture));
    }
    let public = quantization(claims)?;
    if let Some(site) = first_unalike_site(&public) {
        return Err(mismatch(StatementField::Rescaling { site }));
    }
    Ok(public)
}

/// Step 2.1.
fn architecture_holds(arch: &ArchSpec) -> bool {
    // `n_kv_heads ≥ 1` divides `n_heads ≥ 1`.
    let heads_divide = arch.n_heads >= 1
        && arch.n_kv_heads >= 1
        && arch.n_heads.checked_rem(arch.n_kv_heads) == Some(0)
        // RFC-0047: a query-head group has an integral log2, even for odd head counts.
        && arch
            .n_heads
            .checked_div(arch.n_kv_heads)
            .is_some_and(|group| group.count_ones() == 1);
    let head_width = arch.d_head.is_multiple_of(2) && (2..=(1 << 16) - 2).contains(&arch.d_head);
    let width = u64::from(arch.n_heads).checked_mul(u64::from(arch.d_head))
        == Some(u64::from(arch.d_model));
    arch.n_layers >= 1
        && heads_divide
        && head_width
        && width
        && arch.d_ff >= 1
        && (1..=1 << 18).contains(&arch.vocab)
        && (1..=1 << 17).contains(&arch.max_context)
}

/// Step 2.2, which also indexes the data it accepts by layer and site.
fn quantization<'a>(claims: &'a ClaimList) -> Result<PublicPart<'a>, ClaimsError> {
    let refused = || mismatch(StatementField::Quantization);
    let (arch, quant, profile) = (&claims.arch, &claims.quant, claims.profile);
    if quant.rope != arch.rope {
        return Err(refused());
    }
    // One score scale and one SiLU exponent per layer, in order: their lengths bound the layers,
    // so the index below is no larger than the claim list.
    let layers = usize::try_from(arch.n_layers).map_err(|_| refused())?;
    if quant.score_scales.len() != layers || claims.silu_in_exponents.len() != layers {
        return Err(refused());
    }
    let mut parts = Vec::new();
    parts
        .try_reserve_exact(layers)
        .map_err(|_| ClaimsError::Memory)?;
    for (index, (&(scale_layer, scale), &(silu_layer, silu))) in quant
        .score_scales
        .iter()
        .zip(&claims.silu_in_exponents)
        .enumerate()
    {
        let index = u32::try_from(index).map_err(|_| refused())?;
        if scale_layer != index || silu_layer != index || scale.num() <= 0 {
            return Err(refused());
        }
        parts.push(LayerPart {
            rescales: [None; 11],
            exponents: [None; 11],
            score_scale: scale,
            silu_in_exponent: silu,
        });
    }
    // For each layer's `SoftmaxRow`, the first `Exp` width exists and RFC-0008 allows its table
    // (RFC-0008 'Table specifications', RFC-0030 'Range analysis and validation').
    let from = spread_bits(arch.d_head).ok_or(ClaimsError::Internal)?;
    for part in &parts {
        if !admits_exp_table(from, quant.exp_sigma, part.score_scale)? {
            return Err(refused());
        }
    }
    // Every `SoftmaxRow` row has one reciprocal (RFC-0009 Lemma 6).
    if !reciprocal_unique(quant.exp_sigma, arch.max_context) {
        return Err(refused());
    }

    // `act_exponents`: exactly the INT8 points, in order.
    let mut exponents = quant.act_exponents.iter();
    let embed_exponent = match exponents.next() {
        Some(&(QuantPoint::Embed, e)) => e,
        _ => return Err(refused()),
    };
    for (layer, part) in (0u32..).zip(parts.iter_mut()) {
        for site in Site::ALL {
            if !(site.in_profile(profile) && site.is_int8()) {
                continue;
            }
            match exponents.next() {
                Some(&(point, e)) if point == (QuantPoint::Layer { layer, site }) => {
                    let slot = part
                        .exponents
                        .get_mut(usize::from(site.tag()))
                        .ok_or(ClaimsError::Internal)?;
                    *slot = Some(e);
                }
                _ => return Err(refused()),
            }
        }
    }
    if !matches!(exponents.next(), Some((QuantPoint::FinalNorm, _))) || exponents.next().is_some() {
        return Err(refused());
    }

    // `shift_class_sets` and `affine_sites`: exactly the rescale points between them, in order.
    let mut pots = quant.shift_class_sets.iter().peekable();
    let mut affines = quant.affine_sites.iter().peekable();
    let mut next = |point: QuantPoint, site: Option<Site>| -> Result<Rescale<'a>, ClaimsError> {
        let in_pot = pots.peek().is_some_and(|(p, _)| *p == point);
        let in_affine = affines.peek().is_some_and(|(p, _)| *p == point);
        match (in_pot, in_affine) {
            (true, false) => {
                let (_, shifts) = pots.next().ok_or(ClaimsError::Internal)?;
                let follows_weight = site.and_then(Site::weight_before).is_some();
                let classes_hold = (1..=4).contains(&shifts.len())
                    && shifts.iter().all(|r| (1..=63).contains(r))
                    && (follows_weight || shifts.len() == 1);
                if classes_hold {
                    Ok(Rescale::Pot(shifts))
                } else {
                    Err(refused())
                }
            }
            (false, true) => {
                let &(_, r) = affines.next().ok_or(ClaimsError::Internal)?;
                if site.is_some_and(Site::may_be_affine) && (1..=63).contains(&r) {
                    Ok(Rescale::Affine(r))
                } else {
                    Err(refused())
                }
            }
            // In both lists, or in neither: a point out of order or repeated is in neither here.
            _ => Err(refused()),
        }
    };
    for (layer, part) in (0u32..).zip(parts.iter_mut()) {
        for site in Site::ALL {
            if !site.in_profile(profile) {
                continue;
            }
            let rescale = next(QuantPoint::Layer { layer, site }, Some(site))?;
            let slot = part
                .rescales
                .get_mut(usize::from(site.tag()))
                .ok_or(ClaimsError::Internal)?;
            *slot = Some(rescale);
        }
    }
    let final_norm = next(QuantPoint::FinalNorm, None)?;
    if pots.next().is_some() || affines.next().is_some() {
        return Err(refused());
    }

    if !(2..=31).contains(&quant.residual_bits) {
        return Err(refused());
    }
    // The embedding's shift: `127 · 2^{s_e} < 2^{residual_bits − 1}`, and `s_e ≥ 0` so that its
    // constant is an integer. `127 · 2^{s_e} < 2^{b−1}` exactly when `s_e ≤ b − 8`, since
    // `127 < 2^7`.
    let s_e = i16::from(embed_exponent)
        .checked_sub(i16::from(quant.residual_exponent))
        .ok_or(ClaimsError::Internal)?;
    let largest = i16::from(quant.residual_bits)
        .checked_sub(8)
        .ok_or(ClaimsError::Internal)?;
    if !(0..=largest).contains(&s_e) {
        return Err(refused());
    }
    let embed_shift = u32::try_from(s_e).map_err(|_| ClaimsError::Internal)?;
    if quant.rms_eps_int < 1 || quant.rsqrt_kappa > 31 {
        return Err(refused());
    }
    Ok(PublicPart {
        claims,
        layers: parts,
        final_norm,
        embed_shift,
    })
}

/// Whether a layer's score scale admits an `Exp` table: the narrowest `key_bits` from `from`,
/// the spread's width, up to 32 whose `Exp { key_bits, sigma, scale }` maps its last key
/// `2^{key_bits} − 1` to 0 exists, and RFC-0008 allows that table, by
/// [`TableKind::check_parameters`]: `1 ≤ key_bits ≤ 32`, `σ ≤ 32`, `scale > 0`, its zero tail
/// starting by key `2^24`. That is the table the lowering's search ends at (RFC-0044).
fn admits_exp_table(from: u8, sigma: u8, scale: Rational) -> Result<bool, ClaimsError> {
    for key_bits in from..=32 {
        let last = 1u64
            .checked_shl(u32::from(key_bits))
            .and_then(|rows| rows.checked_sub(1))
            .ok_or(ClaimsError::Internal)?;
        // The scales are positive here, so the value test cannot fail.
        if exp_maps_to_zero(last, sigma, scale).map_err(|_| ClaimsError::Internal)? {
            let table = TableKind::Exp {
                key_bits,
                sigma,
                scale,
            };
            return match table.check_parameters() {
                Ok(_) => Ok(true),
                Err(error) if error.is_rejection() => Ok(false),
                Err(TableError::Memory) => Err(ClaimsError::Memory),
                Err(_) => Err(ClaimsError::Internal),
            };
        }
    }
    Ok(false)
}

/// RFC-0009 Lemma 6's precondition, `(2·max_context − 1)·2^{2σ} < P`, exact in 128-bit integers.
/// A row sum is at most `max_context·2^σ`. Below the bound, a reciprocal times the row sum plus
/// its slack stays below `P + 2^{2σ}`, so the bracket admits `⌊2^{2σ}/s_q⌋` alone. Past it, a
/// row of `max_context` positions at its maximal score admits a second reciprocal, whose product
/// wraps past `P`. A shift past 127 bits or a product past `2^128` is past `P` too.
fn reciprocal_unique(sigma: u8, max_context: u32) -> bool {
    let positions = u128::from(max_context)
        .checked_mul(2)
        .and_then(|twice| twice.checked_sub(1));
    let power = u32::from(sigma)
        .checked_mul(2)
        .and_then(|bits| 1u128.checked_shl(bits));
    match (positions, power) {
        (Some(positions), Some(power)) => positions
            .checked_mul(power)
            .is_some_and(|product| product < u128::from(Fp::MODULUS)),
        // No context is 0 once step 2.1 has passed.
        _ => false,
    }
}

/// Step 2.3: the first site, in declaration order, that is affine in some layers only, or a
/// `pot` site of one class in some layers and several in others.
fn first_unalike_site(public: &PublicPart<'_>) -> Option<Site> {
    let shape = |rescale: Option<Rescale<'_>>| match rescale {
        Some(Rescale::Affine(_)) => Some(0),
        Some(Rescale::Pot(&[_])) => Some(1),
        Some(Rescale::Pot(_)) => Some(2),
        None => None,
    };
    Site::ALL.into_iter().find(|&site| {
        let first = shape(public.rescale(0, site));
        (1..public.layers()).any(|layer| shape(public.rescale(layer, site)) != first)
    })
}

// ----- Step 3: the profile -------------------------------------------------------------------

/// `2^31 − 1`, INT32's greatest value.
const INT32_MAX: u128 = 2_147_483_647;

/// S1 to S3 of RFC-0005 'Profiles' on the derived graph, as the module documentation states them.
fn profile_holds(graph: &Graph, profile: Profile) -> Result<bool, ClaimsError> {
    for node in graph.nodes() {
        match *node.op() {
            Op::MatMulWeight { weight } => {
                let (x, w) = match *node.inputs() {
                    [x, w] => (x, w),
                    _ => return Err(ClaimsError::Internal),
                };
                let fan_in = graph
                    .tensor(w)
                    .and_then(|t| t.shape.dims().get(1).copied())
                    .ok_or(ClaimsError::Internal)?;
                let producer = graph
                    .tensor(x)
                    .ok_or(ClaimsError::Internal)?
                    .producer
                    .and_then(|id| graph.node(id))
                    .map(Node::op);
                let (range, one_hot) = match producer {
                    Some(&Op::Requant { lo, hi, .. }) => (Some((lo, hi)), false),
                    Some(Op::OneHot { .. }) => (Some((0, 1)), true),
                    _ => (None, false),
                };
                let exempt = profile == Profile::Minimal
                    && matches!(weight.ty, WeightType::O | WeightType::Down);
                let int8 = matches!(range, Some((lo, hi)) if lo >= -128 && hi <= 127);
                if !exempt && !int8 {
                    return Ok(false);
                }
                if profile == Profile::Servable {
                    let bound = if one_hot {
                        127
                    } else {
                        let (lo, hi) = range.ok_or(ClaimsError::Internal)?;
                        let largest = u128::from(lo.unsigned_abs().max(hi.unsigned_abs()));
                        127u128
                            .checked_mul(largest)
                            .and_then(|b| b.checked_mul(u128::from(fan_in)))
                            .ok_or(ClaimsError::Internal)?
                    };
                    if bound > INT32_MAX {
                        return Ok(false);
                    }
                }
            }
            Op::TableMap { table } => {
                let silu = graph
                    .table(table)
                    .is_some_and(|t| matches!(t.kind, TableKind::Silu { .. }));
                if !silu {
                    return Ok(false);
                }
            }
            _ => {}
        }
    }
    Ok(true)
}

// ----- Step 4: the tables --------------------------------------------------------------------

/// Regenerates each table of the claim list in `TableId` order, the first failure the verdict.
fn regenerate(specs: &[tv_tables::TableSpec]) -> Result<Vec<Table>, ClaimsError> {
    let mut order = Vec::new();
    order
        .try_reserve_exact(specs.len())
        .map_err(|_| ClaimsError::Memory)?;
    order.extend(0..specs.len());
    // By id, and among equal ids by position, so that the order is a function of the list.
    order.sort_unstable_by_key(|&i| (specs.get(i).map(|s| s.id), i));
    let mut tables = Vec::new();
    tables
        .try_reserve_exact(specs.len())
        .map_err(|_| ClaimsError::Memory)?;
    for i in order {
        let spec = specs.get(i).ok_or(ClaimsError::Internal)?;
        match Table::from_spec(spec) {
            Ok(table) => tables.push(table),
            Err(error) if error.is_rejection() => {
                return Err(mismatch(StatementField::Table { table: spec.id }));
            }
            Err(TableError::Memory) => return Err(ClaimsError::Memory),
            Err(_) => return Err(ClaimsError::Internal),
        }
    }
    Ok(tables)
}
