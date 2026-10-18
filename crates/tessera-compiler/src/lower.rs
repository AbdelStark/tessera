// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Llama-3.1 lowering (RFC-0006 'Llama lowering'): the graph of `D` over one sequence of `n`
//! tokens at positions `0` to `n − 1`, RFC-0005 'Integer semantics' step by step.
//!
//! - **Public data only.** [`lower`] reads a [`PublicModel`]: the architecture, the profile and the
//!   quantization data the claim list states. Weights and channel classes enter the graph as
//!   committed inputs, never as values, so the graph, its tables and its ranges are what the
//!   verifier re-derives from the claim list (INV-IR-5, SP-7). [`validate`] checks the rest of an
//!   [`IntegerModel`], its weights and hidden channel classes, against the public part.
//! - **Layout.** Activations are position-major, `[n, channels]`. RoPE rotates the pairs
//!   `(x_i, x_{i + d_head/2})` of each head, the layout of Hugging Face checkpoints. Attention
//!   runs head-major: `Transpose` takes the queries, keys and values from `[n, heads, d_head]` to
//!   `[heads, n, d_head]`, and `Expand` repeats each KV head for its group of query heads, so one
//!   `MatMulAct` forms every head's scores, one causal `SoftmaxRow` their exponentials and
//!   reciprocals, and `Mul(MatMulAct(e, v), R)` the attention output: the pattern the softmax
//!   fact bounds. A last `Transpose` returns the heads to position-major order. Every claim on
//!   these layouts moves coordinates (RFC-0037 'Head layouts').
//! - **The first checkpoint.** The residual that enters the first layer, the embedding scaled by
//!   `2^{s_e}`, is committed advice, which stage 1 proves equal to its `ConstMul`: every layer then
//!   reads a committed residual (RFC-0037 'The committed first checkpoint').
//! - **Residual.** Each update saturates the residual's sum at INT20 (RFC-0022): a site of one
//!   class rescales the fused key `acc + 2^r · res`, a site of several takes two `Requant`s.
//!   Every norm then reads a residual bounded over all inputs.
//! - **Tables.** RoPE's is table 0. Each layer's `Exp` table, for its score scale, and its `Silu`
//!   table follow in order of first use; a layer whose table equals an earlier one shares it.
//!   Their key widths come from public bounds: `Exp` keys cover the scores' spread
//!   `128 · 255 · d_head` of INT8 queries and keys, and `Silu` keys the gate accumulator's
//!   `127 · 128 · d_model`.
//! - **Checks.** Range analysis must accept the graph, which then must keep its profile's
//!   constraints ([`check_profile`]). Every tensor carries its analyzed range and the narrowest
//!   integer type holding it.

use tessera_codec::Rational;
use tessera_ir::{
    ArchSpec, Broadcast, ClaimList, CommitmentClass, Graph, GraphBuilder, IntRange, IntType,
    IntegerModel, IrError, NodeKind, Op, Profile, PublicQuantSpec, QuantPoint, Site, TableId,
    TensorDecl, TensorId, WeightRef, WeightType,
};
use tessera_tables::{GeneratedTable, RopeTable, Table, TableKind, TableSpec};

use crate::{CompileError, Range, Ranges, analyze};

/// The largest vocabulary and context of v1 (RFC-0007 'Verifier validation').
const MAX_VOCAB: u32 = 1 << 18;
const MAX_CONTEXT: u32 = 1 << 17;

/// RoPE's table: the first.
const ROPE: TableId = TableId(0);

/// The saturation range of an INT8 rescale.
const INT8: (i64, i64) = (-128, 127);

/// The public part of `D`: what the claim list states, and all that [`lower`] reads.
#[derive(Clone, Copy, Debug)]
pub struct PublicModel<'a> {
    /// The architecture.
    pub arch: &'a ArchSpec,
    /// The dataflow.
    pub profile: Profile,
    /// The quantization data of the claim list.
    pub quant: &'a PublicQuantSpec,
    /// Each layer's `Silu` input exponent, which the claim list states in the table's
    /// specification.
    pub silu_in_exponents: &'a [(u32, i8)],
}

impl<'a> PublicModel<'a> {
    /// The public part of `model`.
    pub fn of(model: &'a IntegerModel) -> Self {
        Self {
            arch: &model.arch,
            profile: model.profile,
            quant: &model.quant.public,
            silu_in_exponents: &model.quant.silu_in_exponents,
        }
    }

    /// The public part that `claims` states: everything [`lower`] reads, from the claim list
    /// alone, as the verifier lowers it (RFC-0007 'Verifier validation' step 5, RFC-0042).
    pub fn of_claims(claims: &'a ClaimList) -> Self {
        Self {
            arch: &claims.arch,
            profile: claims.profile,
            quant: &claims.quant,
            silu_in_exponents: &claims.silu_in_exponents,
        }
    }
}

/// `D`'s graph over one sequence, the tables it names, and the tensors later stages attach to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lowering {
    /// The graph, every tensor with its analyzed range.
    pub graph: Graph,
    /// The tables its nodes name, by id.
    pub tables: Vec<TableSpec>,
    /// The token ids `[n]`, committed advice: the anchor circuits read the token stream here
    /// (RFC-0015).
    pub tokens: TensorId,
    /// The positions `[n]`, public: `0` to `n − 1`.
    pub positions: TensorId,
    /// The one-hot tokens `[n, vocab]`: scoring at position `p` reuses row `p + 1` (RFC-0009
    /// 'OneHot').
    pub one_hot: TensorId,
    /// The logits `[n, vocab]`, which the scoring certificates read (RFC-0017).
    pub logits: TensorId,
    /// The residual stream `[n, d_model]` at each layer boundary: after the embedding, then after
    /// each layer. These are the checkpoints from which any layer regenerates (RFC-0018 'Witness
    /// engine').
    pub checkpoints: Vec<TensorId>,
    /// Each rescale site's output, in the order the lowering builds them: a `Requant` output, or
    /// the `Silu` table's. A classed residual site's intermediate update is not its output.
    pub sites: Vec<(QuantPoint, TensorId)>,
    /// The channel-class vector of each weight whose site has several classes, `[rows]`: its
    /// `ChannelParams` classes, committed with the weights (RFC-0012).
    pub classes: Vec<(WeightRef, TensorId)>,
    /// The multipliers of each weight before an affine site, `[rows]`: its `ChannelParams`
    /// multipliers, committed with the weights (RFC-0025).
    pub multipliers: Vec<(WeightRef, TensorId)>,
}

/// Checks `model` whole: its public part as [`lower`] does, and its weights, channel classes and
/// multipliers against it.
///
/// # Errors
///
/// [`CompileError::Model`] for the first rule broken: the public rules first, then the weights in
/// order, then the channel parameters.
pub fn validate(model: &IntegerModel) -> Result<(), CompileError> {
    let public = PublicModel::of(model);
    check_public(public)?;
    let expected = weights(&model.arch);
    if model.weights.len() != expected.len() {
        return refuse("the weights are the architecture's, one per type and layer");
    }
    for (weight, &(r, rows, cols)) in model.weights.iter().zip(&expected) {
        if (weight.ty, weight.layer) != (r.ty, r.layer) {
            return refuse("the weights are the architecture's, ordered by (ty, layer)");
        }
        if (weight.rows, weight.cols) != (rows, cols) {
            return refuse("each weight has its architecture's shape");
        }
        if u64::try_from(weight.data.len()).ok() != Some(u64::from(rows) * u64::from(cols)) {
            return refuse("a weight holds rows · cols entries");
        }
        if weight.data.contains(&i8::MIN) {
            return refuse("every weight lies in [−127, 127]");
        }
    }
    let channels = &model.quant.channels;
    let classed = classed(public)?;
    let order = channels.iter().map(|c| (c.ty, c.layer));
    if !order.eq(classed.iter().map(|&(w, ..)| (w.ty, w.layer))) {
        return refuse(
            "a ChannelParams exists for each weight a site of several classes or an affine site \
             rescales, and only for one, sorted by (ty, layer)",
        );
    }
    for (params, &(_, rows, classes)) in channels.iter().zip(&classed) {
        let per_channel = |len: usize| u64::try_from(len).ok() == Some(u64::from(rows));
        match classes {
            Some(count) => {
                if !per_channel(params.class.len()) {
                    return refuse("a ChannelParams holds one class per output channel");
                }
                if params.class.iter().any(|&c| usize::from(c) >= count) {
                    return refuse("a channel class indexes its site's shift classes");
                }
                if !params.multiplier.is_empty() {
                    return refuse("only the weight before an affine site has multipliers");
                }
            }
            // RFC-0025: an affine site's weight carries multipliers, and no classes.
            None => {
                if !per_channel(params.multiplier.len()) {
                    return refuse(
                        "an affine site's ChannelParams holds one multiplier per channel",
                    );
                }
                if params.multiplier.contains(&0) {
                    return refuse("a multiplier lies from 1 to 2^16 − 1");
                }
                if !params.class.is_empty() {
                    return refuse("an affine site's ChannelParams holds no classes");
                }
            }
        }
    }
    Ok(())
}

/// The graph of `D`'s forward pass over `positions` tokens, its tables, and its hooks.
///
/// # Errors
///
/// - [`CompileError::Model`] when the public part breaks a rule of RFC-0005, RFC-0006 'Llama
///   lowering', RFC-0007 'Verifier validation' or RFC-0025; [`CompileError::ModelTable`] when a
///   table it defines is one RFC-0008 refuses;
/// - [`CompileError::Usage`] for no position, or more than `max_context`;
/// - [`CompileError::Range`] and [`CompileError::Wrap`] when range analysis refuses the graph;
/// - [`CompileError::Profile`] when it breaks a constraint of its profile.
pub fn lower(model: PublicModel<'_>, positions: u32) -> Result<Lowering, CompileError> {
    check_public(model)?;
    if positions == 0 || positions > model.arch.max_context {
        return Err(CompileError::Usage {
            rule: "a sequence has 1 to max_context positions",
        });
    }
    // Every row count the attention layout forms, `n_heads · n` at most, fits a dimension.
    if model.arch.n_heads.checked_mul(positions).is_none() {
        return Err(CompileError::Usage {
            rule: "n_heads · positions < 2^32",
        });
    }
    let mut builder = Builder::new(model, positions)?;
    let (one_hot, logits, checkpoints) = builder.forward()?;
    let mut graph = builder.graph.finish();
    let ranges = analyze(&graph, &builder.tables)?;
    check_profile(&graph, &ranges, &builder.tables, model.profile)?;
    settle(&mut graph, &ranges)?;
    Ok(Lowering {
        graph,
        tables: builder.tables,
        tokens: builder.tokens,
        positions: builder.positions,
        one_hot,
        logits,
        checkpoints,
        classes: builder.classes,
        multipliers: builder.multipliers,
        sites: builder.sites,
    })
}

/// Checks the constraints `profile` claims on a forward pass (RFC-0005 'Profiles'), with the
/// ranges range analysis gives it:
///
/// - **S1**: every weight matmul's operands are INT8, except, in `minimal`, the O and down
///   projections';
/// - **S2**, `servable` only: every weight matmul's accumulator fits INT32;
/// - **S3**: every nonlinearity is a lookup into a `Silu` or `Exp` table or a gadget of RFC-0005's
///   semantics: `TableMap` takes only `Silu` tables, and neither the scoring certificates nor the
///   anchors' byte operations are part of `D`.
///
/// The attention products are outside S1 and S2: `PV` multiplies 17-bit exponentials, in both
/// profiles.
///
/// # Errors
///
/// [`CompileError::Profile`] at the first node that breaks one.
pub fn check_profile(
    graph: &Graph,
    ranges: &Ranges,
    tables: &[TableSpec],
    profile: Profile,
) -> Result<(), CompileError> {
    let within = |t: TensorId, lo: i128, hi: i128| {
        ranges
            .of(t)
            .and_then(Range::bounded)
            .is_some_and(|r| lo <= r.lo && r.hi <= hi)
    };
    for node in &graph.nodes {
        let fail = |rule| {
            Err(CompileError::Profile {
                node: node.id,
                rule,
            })
        };
        match &node.op {
            Op::MatMulWeight { weight } => {
                let exempt = profile == Profile::Minimal
                    && matches!(weight.ty, WeightType::O | WeightType::Down);
                if !exempt && !node.inputs.iter().all(|&t| within(t, -128, 127)) {
                    return fail("S1: a weight matmul's operands are INT8");
                }
                let int32 = (i128::from(i32::MIN), i128::from(i32::MAX));
                if profile == Profile::Servable && !within(node.outputs[0], int32.0, int32.1) {
                    return fail("S2: a weight matmul's accumulator fits INT32");
                }
            }
            Op::TableMap { table } => {
                let silu = tables
                    .iter()
                    .any(|s| s.id == *table && matches!(s.kind, TableKind::Silu { .. }));
                if !silu {
                    return fail("S3: an elementwise nonlinearity is the SiLU table");
                }
            }
            Op::Bitwise { .. } | Op::Argmax | Op::InverseCdf { .. } => {
                return fail("S3: a forward pass holds only RFC-0005's operations");
            }
            _ => {}
        }
    }
    Ok(())
}

// ----- Validation -------------------------------------------------------------------------------

fn refuse<T>(rule: &'static str) -> Result<T, CompileError> {
    Err(CompileError::Model { rule })
}

/// The rescale sites of a layer of `profile`, in RFC-0005's order.
fn sites(profile: Profile) -> &'static [Site] {
    match profile {
        Profile::Servable => &[
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
        ],
        Profile::Minimal => &[
            Site::AttnNorm,
            Site::RopeQ,
            Site::RopeK,
            Site::V,
            Site::AttnResidual,
            Site::MlpNorm,
            Site::Silu,
            Site::MlpResidual,
        ],
    }
}

/// The weight whose output channels `site` rescales, if any (03 §2 'Quantization').
fn after(site: Site) -> Option<WeightType> {
    match site {
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

/// The rescale points of a model, sorted: each layer's sites, then the final norm. `int8` keeps
/// only those with an INT8 output and an exponent of their own, and adds the embedding: the
/// residual updates take the residual's. Lazy, so that comparing a short list with the points of
/// a model claiming many layers stops at the list's end.
fn points(arch: &ArchSpec, profile: Profile, int8: bool) -> impl Iterator<Item = QuantPoint> {
    let residual = |site| matches!(site, Site::AttnResidual | Site::MlpResidual);
    let embed = int8.then_some(QuantPoint::Embed);
    let layers = (0..arch.n_layers).flat_map(move |layer| {
        sites(profile)
            .iter()
            .filter(move |&&site| !(int8 && residual(site)))
            .map(move |&site| QuantPoint::Layer { layer, site })
    });
    embed
        .into_iter()
        .chain(layers)
        .chain([QuantPoint::FinalNorm])
}

/// Every weight of `arch` in `(ty, layer)` order, with its rows and columns.
fn weights(arch: &ArchSpec) -> Vec<(WeightRef, u32, u32)> {
    let (d, ff, vocab) = (arch.d_model, arch.d_ff, arch.vocab);
    let (q, kv) = (arch.n_heads * arch.d_head, arch.n_kv_heads * arch.d_head);
    let mut all = vec![(
        WeightRef {
            ty: WeightType::Embed,
            layer: 0,
        },
        d,
        vocab,
    )];
    for (ty, rows, cols) in [
        (WeightType::Q, q, d),
        (WeightType::K, kv, d),
        (WeightType::V, kv, d),
        (WeightType::O, d, q),
        (WeightType::Gate, ff, d),
        (WeightType::Up, ff, d),
        (WeightType::Down, d, ff),
    ] {
        all.extend((0..arch.n_layers).map(|layer| (WeightRef { ty, layer }, rows, cols)));
    }
    all.push((
        WeightRef {
            ty: WeightType::LmHead,
            layer: 0,
        },
        vocab,
        d,
    ));
    all
}

/// Each weight a site of several classes or an affine site rescales, sorted by `(ty, layer)`,
/// with its rows and its site's class count, or `None` for an affine site.
fn classed(model: PublicModel<'_>) -> Result<Vec<(WeightRef, u32, Option<usize>)>, CompileError> {
    let shapes = weights(model.arch);
    let mut classed = Vec::new();
    for layer in 0..model.arch.n_layers {
        for &site in sites(model.profile) {
            let Some(ty) = after(site) else { continue };
            let point = QuantPoint::Layer { layer, site };
            let classes = match affine(model.quant, point) {
                Some(_) => None,
                None => match shifts(model.quant, point)?.len() {
                    1 => continue,
                    count => Some(count),
                },
            };
            let weight = WeightRef { ty, layer };
            let rows = shapes
                .iter()
                .find(|(w, ..)| *w == weight)
                .map_or(0, |&(_, rows, _)| rows);
            classed.push((weight, rows, classes));
        }
    }
    classed.sort_by_key(|&(w, ..)| w);
    Ok(classed)
}

/// The public rules of RFC-0005, RFC-0006 'Llama lowering' and RFC-0007 'Verifier validation'.
fn check_public(model: PublicModel<'_>) -> Result<(), CompileError> {
    let arch = model.arch;
    if arch.n_layers == 0 {
        return refuse("a model has at least one layer");
    }
    if arch.n_kv_heads == 0 || !arch.n_heads.is_multiple_of(arch.n_kv_heads) {
        return refuse("n_kv_heads divides n_heads");
    }
    if arch.d_head < 2 || !arch.d_head.is_multiple_of(2) || arch.d_head > u32::from(u16::MAX) {
        return refuse("d_head is even, from 2 to 2^16 − 2");
    }
    if arch.n_heads.checked_mul(arch.d_head) != Some(arch.d_model) {
        return refuse("d_model = n_heads · d_head");
    }
    if arch.d_ff == 0 {
        return refuse("d_ff ≥ 1");
    }
    if !(1..=MAX_VOCAB).contains(&arch.vocab) {
        return refuse("1 ≤ vocab ≤ 2^18");
    }
    if !(1..=MAX_CONTEXT).contains(&arch.max_context) {
        return refuse("1 ≤ max_context ≤ 2^17");
    }
    let quant = model.quant;
    if quant.rope != arch.rope {
        return refuse("the quantization data's RoPE is the architecture's");
    }
    let act = quant.act_exponents.iter().map(|&(p, _)| p);
    if !act.eq(points(arch, model.profile, true)) {
        return refuse(
            "act_exponents holds the embedding, each INT8 site of the profile and the final \
             norm, once each and sorted",
        );
    }
    // RFC-0025: each rescale site is `pot`, with its classes, or affine, with its shift.
    let pot: Vec<QuantPoint> = quant.shift_class_sets.iter().map(|(p, _)| *p).collect();
    let affine: Vec<QuantPoint> = quant.affine_sites.iter().map(|&(p, _)| p).collect();
    let increasing = |points: &[QuantPoint]| points.windows(2).all(|w| w[0] < w[1]);
    let mut every = [pot.as_slice(), affine.as_slice()].concat();
    every.sort_unstable();
    if !increasing(&pot)
        || !increasing(&affine)
        || !every.into_iter().eq(points(arch, model.profile, false))
    {
        return refuse(SITES);
    }
    for &(point, r) in &quant.affine_sites {
        let after_its_weight = matches!(
            point,
            QuantPoint::Layer {
                site: Site::RopeQ
                    | Site::RopeK
                    | Site::V
                    | Site::AttnResidual
                    | Site::Up
                    | Site::MlpResidual,
                ..
            }
        );
        if !after_its_weight {
            return refuse("an affine site follows the Q, K, V, O, up or down projection");
        }
        if !(1..=63).contains(&r) {
            return refuse("an affine site's shift lies from 1 to 63");
        }
    }
    for (point, shifts) in &quant.shift_class_sets {
        if !(1..=4).contains(&shifts.len()) || shifts.iter().any(|r| !(1..=63).contains(r)) {
            return refuse("a site has one to four classes, each shift from 1 to 63");
        }
        let weighted = matches!(point, QuantPoint::Layer { site, .. } if after(*site).is_some());
        if !weighted && shifts.len() > 1 {
            return refuse("a site after no weight has one class");
        }
    }
    let layers = || 0..arch.n_layers;
    // RFC-0038: every layer rescales each site alike, so that the layers lower to one template.
    for &site in sites(model.profile) {
        let at = |layer| QuantPoint::Layer { layer, site };
        let affine_in = |layer| self::affine(quant, at(layer)).is_some();
        if layers().any(|layer| affine_in(layer) != affine_in(0)) {
            return refuse("each site is affine in every layer or in none");
        }
        let several = |layer| shifts(quant, at(layer)).is_ok_and(|s| s.len() > 1);
        if !affine_in(0) && layers().any(|layer| several(layer) != several(0)) {
            return refuse("each pot site has one class in every layer, or several in every layer");
        }
    }
    if !quant.score_scales.iter().map(|&(l, _)| l).eq(layers()) {
        return refuse("score_scales holds one scale per layer, sorted");
    }
    if quant.score_scales.iter().any(|(_, scale)| scale.num() < 1) {
        return refuse("a score scale is positive");
    }
    if !model.silu_in_exponents.iter().map(|&(l, _)| l).eq(layers()) {
        return refuse("silu_in_exponents holds one exponent per layer, sorted");
    }
    // A saturating residual sum's update clamps within `±(2^{bits} − 1)`, a table's 32-bit value.
    if !(2..=31).contains(&quant.residual_bits) {
        return refuse("the residual has 2 to 31 bits");
    }
    // 127 · 2^{s_e} < 2^{bits−1}: s_e ≤ 23 at 31 bits, so a shift of 64 or more fails anyway.
    let shift = embed_shift(quant)?;
    let inside = u32::try_from(shift)
        .ok()
        .filter(|&s| s < 64)
        .is_some_and(|s| (127i128 << s) < 1i128 << (quant.residual_bits - 1));
    if !inside {
        return refuse("the embedding's shift into the residual keeps INT8 rows inside it");
    }
    if quant.rms_eps_int == 0 {
        return refuse("RMSNorm's ε term e is at least 1, so that ρ exists");
    }
    if quant.rsqrt_kappa > 31 {
        return refuse("κ ≤ 31, so that N = d · 2^{2κ} stays below 2^94");
    }
    Ok(())
}

/// `s_e = e_Embed − residual_exponent` (03 §2 'Quantization'), before its range is checked.
fn embed_shift(quant: &PublicQuantSpec) -> Result<i64, CompileError> {
    let embed = exponent(quant, QuantPoint::Embed)?;
    Ok(i64::from(embed) - i64::from(quant.residual_exponent))
}

fn exponent(quant: &PublicQuantSpec, point: QuantPoint) -> Result<i8, CompileError> {
    quant
        .act_exponents
        .iter()
        .find(|&&(p, _)| p == point)
        .map(|&(_, e)| e)
        .ok_or(CompileError::Model {
            rule: "act_exponents holds the embedding, each INT8 site of the profile and the final \
                   norm, once each and sorted",
        })
}

fn shifts(quant: &PublicQuantSpec, point: QuantPoint) -> Result<Vec<u8>, CompileError> {
    let set = quant
        .shift_class_sets
        .iter()
        .find(|(p, _)| *p == point)
        .ok_or(CompileError::Model { rule: SITES })?;
    set.1
        .iter()
        .map(|&r| u8::try_from(r))
        .collect::<Result<_, _>>()
        .map_err(|_| CompileError::Model {
            rule: "a site has one to four classes, each shift from 1 to 63",
        })
}

/// The rule that every rescale site has its data once (RFC-0007 'Schema', RFC-0025).
const SITES: &str = "shift_class_sets and affine_sites hold each site of the profile and the final \
                     norm, once each in one of them, each sorted";

/// The shift of the affine site at `point`, if it is one (RFC-0025).
fn affine(quant: &PublicQuantSpec, point: QuantPoint) -> Option<i8> {
    quant
        .affine_sites
        .iter()
        .find(|&&(p, _)| p == point)
        .map(|&(_, r)| r)
}

/// The number of bits of `x`: `2^{bits − 1} ≤ x < 2^bits`, and 0 for 0.
fn bits(x: u128) -> u8 {
    // At most 128.
    (u128::BITS - x.leading_zeros()) as u8
}

// ----- The graph ------------------------------------------------------------------------------

struct Builder<'a> {
    model: PublicModel<'a>,
    n: u32,
    graph: GraphBuilder,
    tables: Vec<TableSpec>,
    classes: Vec<(WeightRef, TensorId)>,
    multipliers: Vec<(WeightRef, TensorId)>,
    /// Each rescale site's output, in building order.
    sites: Vec<(QuantPoint, TensorId)>,
    tokens: TensorId,
    positions: TensorId,
}

fn decl(shape: &[u32], kind: NodeKind) -> TensorDecl {
    // `settle` replaces the type and range with the analysis's.
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType::I8,
        range: IntRange { lo: 0, hi: 0 },
        kind,
    }
}

fn virt(shape: &[u32]) -> TensorDecl {
    decl(shape, NodeKind::Virtual)
}

fn advice(shape: &[u32]) -> TensorDecl {
    decl(
        shape,
        NodeKind::Committed {
            commitment: CommitmentClass::Advice,
        },
    )
}

/// An input of `[0, last]`, typed as narrowly as that allows.
fn whole(shape: &[u32], last: u32, kind: NodeKind) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType {
            bits: bits(u128::from(last)).max(1),
            signed: false,
        },
        range: IntRange {
            lo: 0,
            hi: i128::from(last),
        },
        kind,
    }
}

impl<'a> Builder<'a> {
    fn new(model: PublicModel<'a>, n: u32) -> Result<Self, CompileError> {
        let arch = model.arch;
        let mut graph = GraphBuilder::new();
        let tokens = graph.input(whole(
            &[n],
            arch.vocab - 1,
            NodeKind::Committed {
                commitment: CommitmentClass::Advice,
            },
        ))?;
        let positions = graph.input(whole(&[n], n - 1, NodeKind::Public))?;
        let rope = arch.rope;
        let kind = TableKind::Rope {
            theta: rope.theta,
            factor: rope.factor,
            low_freq_factor: rope.low_freq_factor,
            high_freq_factor: rope.high_freq_factor,
            original_max: rope.original_max,
            bits: rope.bits,
            max_context: arch.max_context,
            // At most 2^16 − 2, checked with the architecture.
            d_head: u16::try_from(arch.d_head).unwrap_or(u16::MAX),
        };
        let rope = RopeTable::new(ROPE, kind)
            .map_err(|error| CompileError::ModelTable { table: ROPE, error })?;
        Ok(Self {
            model,
            n,
            graph,
            tables: vec![rope.spec().clone()],
            classes: Vec::new(),
            multipliers: Vec::new(),
            sites: Vec::new(),
            tokens,
            positions,
        })
    }

    fn node(
        &mut self,
        op: Op,
        inputs: &[TensorId],
        output: TensorDecl,
    ) -> Result<TensorId, IrError> {
        Ok(self.graph.node(op, inputs, vec![output])?[0])
    }

    /// The forward pass: the one-hot tokens, the logits and the residual at each layer boundary.
    fn forward(&mut self) -> Result<(TensorId, TensorId, Vec<TensorId>), CompileError> {
        let arch = self.model.arch;
        let (n, d, vocab) = (self.n, arch.d_model, arch.vocab);
        // Embedding: the one-hot tokens select columns of the transposed embedding, shifted into
        // the residual's scale.
        let one_hot = self.node(Op::OneHot { vocab }, &[self.tokens], advice(&[n, vocab]))?;
        let x0 = self.matmul(WeightType::Embed, 0, one_hot, d, vocab)?;
        let scale = self
            .graph
            .constant(vec![1 << embed_shift(self.model.quant)?])?;
        // The first checkpoint is committed: every layer reads a committed residual (RFC-0037).
        let mut residual = self.node(Op::ConstMul { c: scale }, &[x0], advice(&[n, d]))?;
        let mut checkpoints = vec![residual];
        for layer in 0..arch.n_layers {
            residual = self.layer(layer, residual)?;
            checkpoints.push(residual);
        }
        let y = self.norm(residual, QuantPoint::FinalNorm)?;
        let logits = self.matmul(WeightType::LmHead, 0, y, vocab, d)?;
        Ok((one_hot, logits, checkpoints))
    }

    fn layer(&mut self, layer: u32, residual: TensorId) -> Result<TensorId, CompileError> {
        let arch = self.model.arch;
        let (n, d, ff) = (self.n, arch.d_model, arch.d_ff);
        let (q_width, kv_width) = (arch.n_heads * arch.d_head, arch.n_kv_heads * arch.d_head);
        let point = |site| QuantPoint::Layer { layer, site };
        let servable = self.model.profile == Profile::Servable;

        // Attention.
        let y = self.norm(residual, point(Site::AttnNorm))?;
        let q = self.matmul(WeightType::Q, layer, y, q_width, d)?;
        let k = self.matmul(WeightType::K, layer, y, kv_width, d)?;
        let v = self.matmul(WeightType::V, layer, y, kv_width, d)?;
        let q = self.rope(q, arch.n_heads)?;
        let k = self.rope(k, arch.n_kv_heads)?;
        let q = self.rescale(q, point(Site::RopeQ), q_width, INT8)?;
        let k = self.rescale(k, point(Site::RopeK), kv_width, INT8)?;
        let v = self.rescale(v, point(Site::V), kv_width, INT8)?;
        let mut o = self.attention(layer, q, k, v)?;
        if servable {
            o = self.rescale(o, point(Site::AttnOut), q_width, INT8)?;
        }
        let acc = self.matmul(WeightType::O, layer, o, d, q_width)?;
        let residual = self.residual(acc, residual, point(Site::AttnResidual))?;

        // MLP.
        let y = self.norm(residual, point(Site::MlpNorm))?;
        let gate = self.matmul(WeightType::Gate, layer, y, ff, d)?;
        let up = self.matmul(WeightType::Up, layer, y, ff, d)?;
        let a = self.silu(layer, gate)?;
        let h = if servable {
            let up = self.rescale(up, point(Site::Up), ff, INT8)?;
            let product = self.node(
                Op::Mul {
                    broadcast: Broadcast::Trailing,
                },
                &[a, up],
                virt(&[n, ff]),
            )?;
            self.rescale(product, point(Site::Product), ff, INT8)?
        } else {
            self.node(
                Op::Mul {
                    broadcast: Broadcast::Trailing,
                },
                &[a, up],
                virt(&[n, ff]),
            )?
        };
        let acc = self.matmul(WeightType::Down, layer, h, d, ff)?;
        self.residual(acc, residual, point(Site::MlpResidual))
    }

    /// The saturating sum `clamp(res + ⌊(acc + 2^{r−1}) / 2^r⌋)` of the residual and the
    /// accumulator at `point`, within `[−2^{bits−1}, 2^{bits−1} − 1]` (RFC-0022). At an affine
    /// site the key is `acc · m` with one shift (RFC-0025):
    ///
    /// - a site of one shift in every layer, with `2^r` a constant in each, rescales the fused key
    ///   `acc + 2^r · res`, whose rounding is the accumulator's since `2^r · res` is a multiple of
    ///   `2^r`;
    /// - otherwise, with hidden class shifts or `r = 63` in some layer: `u = rq(acc, r_c)` within
    ///   `±(2^{bits} − 1)`, then `rq(2(res + u), 1)` clamps the sum. The first clamp binds only
    ///   where the sum saturates anyway, so a layer of one shift computes what the fused key
    ///   would. Every layer takes one route, so that the layers lower alike (RFC-0038).
    fn residual(
        &mut self,
        acc: TensorId,
        residual: TensorId,
        point: QuantPoint,
    ) -> Result<TensorId, CompileError> {
        let (n, d) = (self.n, self.model.arch.d_model);
        let bits = self.model.quant.residual_bits;
        let half = 1i64 << (bits - 1);
        let (acc, shifts, classes) = self.key(acc, point, d)?;
        if self.fused(point)
            && let [r] = shifts[..]
            && let Some(power) = 1i64.checked_shl(u32::from(r)).filter(|&p| p > 0)
        {
            let power = self.graph.constant(vec![power])?;
            let shifted = self.node(Op::ConstMul { c: power }, &[residual], virt(&[n, d]))?;
            let key = self.node(Op::Add, &[acc, shifted], virt(&[n, d]))?;
            let shift = self.graph.shift_classes(vec![r])?;
            let op = Op::Requant {
                shift,
                lo: -half,
                hi: half - 1,
            };
            let output = self.node(op, &[key], advice(&[n, d]))?;
            self.sites.push((point, output));
            return Ok(output);
        }
        let update = self.requant(acc, shifts, classes, d, (1 - 2 * half, 2 * half - 1))?;
        let sum = self.node(Op::Add, &[residual, update], virt(&[n, d]))?;
        let two = self.graph.constant(vec![2])?;
        let doubled = self.node(Op::ConstMul { c: two }, &[sum], virt(&[n, d]))?;
        let shift = self.graph.shift_classes(vec![1])?;
        let op = Op::Requant {
            shift,
            lo: -half,
            hi: half - 1,
        };
        let output = self.node(op, &[doubled], advice(&[n, d]))?;
        self.sites.push((point, output));
        Ok(output)
    }

    /// Whether the residual site of `point` takes the fused key, in every layer alike
    /// (RFC-0038): it has one shift, of one class or affine, at most 62, in each.
    fn fused(&self, point: QuantPoint) -> bool {
        let QuantPoint::Layer { site, .. } = point else {
            return false;
        };
        let quant = self.model.quant;
        (0..self.model.arch.n_layers).all(|layer| {
            let at = QuantPoint::Layer { layer, site };
            match affine(quant, at) {
                Some(r) => (1..=62).contains(&r),
                None => matches!(shifts(quant, at).as_deref(), Ok([r]) if *r <= 62),
            }
        })
    }

    /// `W x` for the weight `(ty, layer)` of `rows × cols`, declared here.
    fn matmul(
        &mut self,
        ty: WeightType,
        layer: u32,
        x: TensorId,
        rows: u32,
        cols: u32,
    ) -> Result<TensorId, CompileError> {
        let weight = WeightRef { ty, layer };
        let w = self.graph.weight(weight, rows, cols)?;
        Ok(self.node(Op::MatMulWeight { weight }, &[x, w], virt(&[self.n, rows]))?)
    }

    /// RMSNorm without its gain, folded into the next weights: `y = rq(res · ρ, r)` with
    /// `ρ = max{ρ : ρ^2 (Σ res_i^2 + e) ≤ d · 2^{2κ}}`.
    fn norm(&mut self, residual: TensorId, point: QuantPoint) -> Result<TensorId, CompileError> {
        let quant = self.model.quant;
        let (n, d) = (self.n, self.model.arch.d_model);
        let squares = self.node(Op::SumSquares { axis: 1 }, &[residual], virt(&[n]))?;
        let rsqrt = Op::RsqrtRow {
            kappa: quant.rsqrt_kappa,
            eps_int: quant.rms_eps_int,
            dim: d,
        };
        let rho = self.node(rsqrt, &[squares], advice(&[n]))?;
        let scaled = self.node(
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            &[residual, rho],
            virt(&[n, d]),
        )?;
        self.rescale(scaled, point, d, INT8)
    }

    /// RoPE on `[n, heads · d_head]`, head by head.
    fn rope(&mut self, x: TensorId, heads: u32) -> Result<TensorId, CompileError> {
        let (n, dh) = (self.n, self.model.arch.d_head);
        let split = self.node(Op::Reshape, &[x], virt(&[n, heads, dh]))?;
        let rotated = self.node(
            Op::Rotate { rope: ROPE },
            &[split, self.positions],
            virt(&[n, heads, dh]),
        )?;
        Ok(self.node(Op::Reshape, &[rotated], virt(&[n, heads * dh]))?)
    }

    /// [`requantize`](Self::requantize), as the output of the site at `point`.
    fn rescale(
        &mut self,
        x: TensorId,
        point: QuantPoint,
        channels: u32,
        range: (i64, i64),
    ) -> Result<TensorId, CompileError> {
        let output = self.requantize(x, point, channels, range)?;
        self.sites.push((point, output));
        Ok(output)
    }

    /// `rq` at `point` on `[n, channels]`, saturating to `range`: [`key`](Self::key)'s key and
    /// shifts.
    fn requantize(
        &mut self,
        x: TensorId,
        point: QuantPoint,
        channels: u32,
        range: (i64, i64),
    ) -> Result<TensorId, CompileError> {
        let (key, shifts, classes) = self.key(x, point, channels)?;
        self.requant(key, shifts, classes, channels, range)
    }

    /// What the site at `point` rescales of `x [n, channels]`, with its shifts and its
    /// channel-class vector:
    ///
    /// - at a `pot` site, `x` itself, with the site's class shifts and, when it has several, the
    ///   classes of the weight before it;
    /// - at an affine site, `x · m`, `m` the weight's multipliers broadcast over the positions,
    ///   with its one shift (RFC-0025).
    fn key(
        &mut self,
        x: TensorId,
        point: QuantPoint,
        channels: u32,
    ) -> Result<(TensorId, Vec<u8>, Option<TensorId>), CompileError> {
        if let Some(r) = affine(self.model.quant, point) {
            // From 1 to 63, checked with the public rules.
            let r = u8::try_from(r).unwrap_or(u8::MAX);
            let m = self.multiplier_vector(point, channels)?;
            let mul = Op::Mul {
                broadcast: Broadcast::Leading,
            };
            let key = self.node(mul, &[x, m], virt(&[self.n, channels]))?;
            return Ok((key, vec![r], None));
        }
        let shifts = shifts(self.model.quant, point)?;
        let classes = match shifts.len() {
            1 => None,
            count => Some(self.class_vector(point, channels, count)?),
        };
        Ok((x, shifts, classes))
    }

    /// `rq(key, r_c, lo, hi)` on `[n, channels]`, `r_c` the class's shift among `shifts`.
    fn requant(
        &mut self,
        key: TensorId,
        shifts: Vec<u8>,
        classes: Option<TensorId>,
        channels: u32,
        (lo, hi): (i64, i64),
    ) -> Result<TensorId, CompileError> {
        let shift = self.graph.shift_classes(shifts)?;
        let mut inputs = vec![key];
        inputs.extend(classes);
        Ok(self.node(
            Op::Requant { shift, lo, hi },
            &inputs,
            advice(&[self.n, channels]),
        )?)
    }

    /// The committed multipliers of the output channels of the weight before the affine site
    /// `point`, each in `[0, 2^{16} − 1]` (RFC-0025).
    fn multiplier_vector(
        &mut self,
        point: QuantPoint,
        channels: u32,
    ) -> Result<TensorId, CompileError> {
        let weight = self.weight_before(point)?;
        let kind = NodeKind::Committed {
            commitment: CommitmentClass::Weight,
        };
        let multipliers = self
            .graph
            .input(whole(&[channels], u32::from(u16::MAX), kind))?;
        self.multipliers.push((weight, multipliers));
        Ok(multipliers)
    }

    /// The weight whose output channels the site at `point` rescales.
    fn weight_before(&self, point: QuantPoint) -> Result<WeightRef, CompileError> {
        match point {
            QuantPoint::Layer { layer, site } => after(site).map(|ty| WeightRef { ty, layer }),
            QuantPoint::Embed | QuantPoint::FinalNorm => None,
        }
        .ok_or(CompileError::Model {
            rule: "a site after no weight has one class",
        })
    }

    /// The committed classes of the output channels of the weight before `point`.
    fn class_vector(
        &mut self,
        point: QuantPoint,
        channels: u32,
        count: usize,
    ) -> Result<TensorId, CompileError> {
        let weight = self.weight_before(point)?;
        // One to four classes, checked with the public rules.
        let last = u32::try_from(count - 1).unwrap_or(0);
        let kind = NodeKind::Committed {
            commitment: CommitmentClass::Weight,
        };
        let classes = self.graph.input(whole(&[channels], last, kind))?;
        self.classes.push((weight, classes));
        Ok(classes)
    }

    /// The scores, exponentials and attention output of every head, position-major `[n, d]`.
    fn attention(
        &mut self,
        layer: u32,
        q: TensorId,
        k: TensorId,
        v: TensorId,
    ) -> Result<TensorId, CompileError> {
        let arch = self.model.arch;
        let (n, h, kv, dh) = (self.n, arch.n_heads, arch.n_kv_heads, arch.d_head);
        let q = self.heads(q, h)?;
        let k = self.heads(k, kv)?;
        let v = self.heads(v, kv)?;
        let scores = self.node(
            Op::MatMulAct {
                transpose_rhs: true,
            },
            &[q, k],
            virt(&[h, n, n]),
        )?;
        let quant = self.model.quant;
        let scale = quant
            .score_scales
            .iter()
            .find(|&&(l, _)| l == layer)
            .map(|&(_, scale)| scale)
            .ok_or(CompileError::Model {
                rule: "score_scales holds one scale per layer, sorted",
            })?;
        // INT8 queries and keys give scores in [−128 · 127 · d_head, 128 · 128 · d_head].
        let spread = 128 * 255 * u128::from(dh);
        let exp = self.exp_table(bits(spread), quant.exp_sigma, scale)?;
        let softmax = Op::SoftmaxRow {
            exp_table: exp,
            sigma: quant.exp_sigma,
        };
        let outputs = vec![
            decl(&[h, n, n], NodeKind::LookupOutput { table: exp }),
            advice(&[h, n]),
        ];
        let softmax = self.graph.node(softmax, &[scores], outputs)?;
        let (e, r) = (softmax[0], softmax[1]);
        let pv = self.node(
            Op::MatMulAct {
                transpose_rhs: false,
            },
            &[e, v],
            virt(&[h, n, dh]),
        )?;
        let o = self.node(
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            &[pv, r],
            virt(&[h, n, dh]),
        )?;
        let swapped = self.node(
            Op::Transpose {
                perm: vec![1, 0, 2],
            },
            &[o],
            virt(&[n, h, dh]),
        )?;
        Ok(self.node(Op::Reshape, &[swapped], virt(&[n, h * dh]))?)
    }

    /// `[n, heads · d_head]` laid out head-major as `[n_heads, n, d_head]`: split into heads,
    /// transposed, and, for `heads` KV heads, each repeated for its `n_heads / heads` query heads
    /// (RFC-0037 'Head layouts').
    fn heads(&mut self, x: TensorId, heads: u32) -> Result<TensorId, CompileError> {
        let arch = self.model.arch;
        let (n, h, dh) = (self.n, arch.n_heads, arch.d_head);
        let split = self.node(Op::Reshape, &[x], virt(&[n, heads, dh]))?;
        let major = self.node(
            Op::Transpose {
                perm: vec![1, 0, 2],
            },
            &[split],
            virt(&[heads, n, dh]),
        )?;
        if heads == h {
            return Ok(major);
        }
        let group = h / heads;
        let repeated = self.node(
            Op::Expand {
                axis: 1,
                len: group,
            },
            &[major],
            virt(&[heads, group, n, dh]),
        )?;
        Ok(self.node(Op::Reshape, &[repeated], virt(&[h, n, dh]))?)
    }

    /// `a = T_silu[g]`, with the gate's channel classes when the site has several.
    fn silu(&mut self, layer: u32, gate: TensorId) -> Result<TensorId, CompileError> {
        let arch = self.model.arch;
        let (n, d, ff) = (self.n, arch.d_model, arch.d_ff);
        let point = QuantPoint::Layer {
            layer,
            site: Site::Silu,
        };
        let shifts = shifts(self.model.quant, point)?;
        let count = shifts.len();
        let in_exp = self
            .model
            .silu_in_exponents
            .iter()
            .find(|&&(l, _)| l == layer)
            .map(|&(_, e)| e)
            .ok_or(CompileError::Model {
                rule: "silu_in_exponents holds one exponent per layer, sorted",
            })?;
        // The gate accumulator of INT8 activations lies within ±127 · 128 · d.
        let bound = 127 * 128 * u128::from(d);
        let table = self.table(TableKind::Silu {
            b: bits(bound) + 1,
            r_by_class: shifts,
            in_exp,
            out_exp: exponent(self.model.quant, point)?,
        })?;
        let mut inputs = vec![gate];
        if count > 1 {
            inputs.push(self.class_vector(point, ff, count)?);
        }
        let output = self.node(
            Op::TableMap { table },
            &inputs,
            decl(&[n, ff], NodeKind::LookupOutput { table }),
        )?;
        self.sites.push((point, output));
        Ok(output)
    }

    /// `SoftmaxRow`'s `Exp` table: the narrowest from the spread's width up whose last key maps
    /// to 0, the key every masked position looks up (RFC-0030). Values do not depend on the
    /// width and the zero tail is a suffix, so the first width that holds a zero is the rule's.
    /// RFC-0008 refuses a zero tail that starts past key `2^24`, so the search never widens a
    /// table past 25 bits.
    fn exp_table(
        &mut self,
        spread_bits: u8,
        sigma: u8,
        scale: Rational,
    ) -> Result<TableId, CompileError> {
        let id = TableId(u32::try_from(self.tables.len()).unwrap_or(u32::MAX));
        for key_bits in spread_bits..=32 {
            let kind = TableKind::Exp {
                key_bits,
                sigma,
                scale,
            };
            // Only this function names an `Exp` table, so an earlier one holds the zero.
            if let Some(spec) = self.tables.iter().find(|s| s.kind == kind) {
                return Ok(spec.id);
            }
            let table = GeneratedTable::new(id, kind)
                .map_err(|error| CompileError::ModelTable { table: id, error })?;
            if table.lookup((1 << key_bits) - 1, 0) == Some(0) {
                self.tables.push(table.spec().clone());
                return Ok(id);
            }
        }
        Err(CompileError::Model {
            rule: "an Exp table holds a zero within 2^32 keys",
        })
    }

    /// The table of `kind`: an earlier one of the same definition, or a new one.
    fn table(&mut self, kind: TableKind) -> Result<TableId, CompileError> {
        if let Some(spec) = self.tables.iter().find(|s| s.kind == kind) {
            return Ok(spec.id);
        }
        let id = TableId(u32::try_from(self.tables.len()).unwrap_or(u32::MAX));
        let table = GeneratedTable::new(id, kind)
            .map_err(|error| CompileError::ModelTable { table: id, error })?;
        self.tables.push(table.spec().clone());
        Ok(id)
    }
}

/// Gives every tensor a node produced its analyzed range and the narrowest type holding it.
fn settle(graph: &mut Graph, ranges: &Ranges) -> Result<(), CompileError> {
    let produced: Vec<TensorId> = graph
        .nodes
        .iter()
        .flat_map(|node| node.outputs.iter().copied())
        .collect();
    for tensor in produced {
        let range = ranges.of(tensor).and_then(Range::bounded);
        let typed = range.and_then(|r| narrowest(r).map(|ty| (r, ty)));
        let (range, ty) = typed.ok_or(CompileError::Graph(IrError::Tensor {
            tensor,
            rule: "a lowered tensor has a bound of at most 64 bits",
        }))?;
        let spec = &mut graph.tensors[tensor.0 as usize];
        spec.range = range;
        spec.ty = ty;
    }
    Ok(())
}

/// The narrowest integer type holding `range`, if one of at most 64 bits does.
fn narrowest(range: IntRange) -> Option<IntType> {
    let width = |x: i128| bits(u128::try_from(x).unwrap_or(0));
    let (bits, signed) = if range.lo >= 0 {
        (width(range.hi).max(1), false)
    } else {
        // `[−2^{b−1}, 2^{b−1} − 1]` holds it when `2^{b−1} > −lo − 1 = !lo` and `2^{b−1} > hi`.
        (1 + width(!range.lo).max(width(range.hi)), true)
    };
    (bits <= 64).then_some(IntType { bits, signed })
}
