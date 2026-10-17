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
//!   runs head-major: `GatherPublic` reorders the queries' rows to `[n_heads, n, d_head]`, and the
//!   keys' and values' to the same shape with each KV head repeated for its group of queries, so
//!   one `MatMulAct` forms every head's scores, one causal `SoftmaxRow` their exponentials and
//!   reciprocals, and `Mul(MatMulAct(e, v), R)` the attention output: the pattern the softmax
//!   fact bounds. A last gather returns the heads to position-major order.
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

use tessera_ir::{
    ArchSpec, CommitmentClass, ConstRef, Graph, GraphBuilder, IntRange, IntType, IntegerModel,
    IrError, NodeKind, Op, Profile, PublicQuantSpec, QuantPoint, Site, TableId, TensorDecl,
    TensorId, WeightRef, WeightType,
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
    /// The channel-class vector of each weight whose site has several classes, `[rows]`: its
    /// `ChannelParams` classes, committed with the weights (RFC-0012).
    pub classes: Vec<(WeightRef, TensorId)>,
}

/// Checks `model` whole: its public part as [`lower`] does, and its weights and channel classes
/// against it.
///
/// # Errors
///
/// [`CompileError::Model`] for the first rule broken: the public rules first, then the weights in
/// order, then the channel classes.
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
    if channels.iter().any(|c| !c.multiplier.is_empty()) {
        return refuse("an affine rescale needs public data the claim list lacks yet (#274)");
    }
    let classed = classed(public)?;
    let order = channels.iter().map(|c| (c.ty, c.layer));
    if !order.eq(classed.iter().map(|&(w, ..)| (w.ty, w.layer))) {
        return refuse(
            "a ChannelParams exists for each weight a site of several classes rescales, and \
             only for one, sorted by (ty, layer)",
        );
    }
    for (params, &(_, rows, count)) in channels.iter().zip(&classed) {
        if u64::try_from(params.class.len()).ok() != Some(u64::from(rows)) {
            return refuse("a ChannelParams holds one class per output channel");
        }
        if params.class.iter().any(|&c| usize::from(c) >= count) {
            return refuse("a channel class indexes its site's shift classes");
        }
    }
    Ok(())
}

/// The graph of `D`'s forward pass over `positions` tokens, its tables, and its hooks.
///
/// # Errors
///
/// - [`CompileError::Model`] when the public part breaks a rule of RFC-0005, RFC-0006 'Llama
///   lowering' or RFC-0007 'Verifier validation', and [`CompileError::ModelTable`] when a table it
///   defines is one RFC-0008 refuses;
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
    let (one_hot, logits) = builder.forward()?;
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
        classes: builder.classes,
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

/// Each weight a site of several classes rescales, sorted by `(ty, layer)`, with its rows and
/// its site's class count.
fn classed(model: PublicModel<'_>) -> Result<Vec<(WeightRef, u32, usize)>, CompileError> {
    let shapes = weights(model.arch);
    let mut classed = Vec::new();
    for layer in 0..model.arch.n_layers {
        for &site in sites(model.profile) {
            let Some(ty) = after(site) else { continue };
            let count = shifts(model.quant, QuantPoint::Layer { layer, site })?.len();
            if count > 1 {
                let weight = WeightRef { ty, layer };
                let rows = shapes
                    .iter()
                    .find(|(w, ..)| *w == weight)
                    .map_or(0, |&(_, rows, _)| rows);
                classed.push((weight, rows, count));
            }
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
    let sets = quant.shift_class_sets.iter().map(|(p, _)| *p);
    if !sets.eq(points(arch, model.profile, false)) {
        return refuse(
            "shift_class_sets holds each site of the profile and the final norm, once each and \
             sorted",
        );
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
        .ok_or(CompileError::Model {
            rule: "shift_class_sets holds each site of the profile and the final norm, once each \
                   and sorted",
        })?;
    set.1
        .iter()
        .map(|&r| u8::try_from(r))
        .collect::<Result<_, _>>()
        .map_err(|_| CompileError::Model {
            rule: "a site has one to four classes, each shift from 1 to 63",
        })
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
    tokens: TensorId,
    positions: TensorId,
    /// Gathers the queries' rows `[n · n_heads, d_head]` into head-major order.
    queries: ConstRef,
    /// Gathers the keys' or values' rows `[n · n_kv_heads, d_head]` into head-major order, each KV
    /// head once per query head of its group.
    keys: ConstRef,
    /// Gathers head-major rows `[n_heads · n, d_head]` back into position-major order.
    back: ConstRef,
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
        // Row `head · n + p` of the head-major layout is row `p · heads + head` of the
        // position-major one, with the KV head `head / group` for keys and values.
        let (h, kv) = (i64::from(arch.n_heads), i64::from(arch.n_kv_heads));
        let (group, count) = (h / kv, i64::from(n));
        let queries = (0..h)
            .flat_map(|head| (0..count).map(move |p| p * h + head))
            .collect();
        let keys = (0..h)
            .flat_map(|head| (0..count).map(move |p| p * kv + head / group))
            .collect();
        let back = (0..count)
            .flat_map(|p| (0..h).map(move |head| head * count + p))
            .collect();
        let queries = graph.constant(queries)?;
        let keys = graph.constant(keys)?;
        let back = graph.constant(back)?;
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
            tokens,
            positions,
            queries,
            keys,
            back,
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

    /// The forward pass: the one-hot tokens and the logits.
    fn forward(&mut self) -> Result<(TensorId, TensorId), CompileError> {
        let arch = self.model.arch;
        let (n, d, vocab) = (self.n, arch.d_model, arch.vocab);
        // Embedding: the one-hot tokens select columns of the transposed embedding, shifted into
        // the residual's scale.
        let one_hot = self.node(Op::OneHot { vocab }, &[self.tokens], advice(&[n, vocab]))?;
        let x0 = self.matmul(WeightType::Embed, 0, one_hot, d, vocab)?;
        let scale = self
            .graph
            .constant(vec![1 << embed_shift(self.model.quant)?])?;
        let mut residual = self.node(Op::ConstMul { c: scale }, &[x0], virt(&[n, d]))?;
        for layer in 0..arch.n_layers {
            residual = self.layer(layer, residual)?;
        }
        let y = self.norm(residual, QuantPoint::FinalNorm)?;
        let logits = self.matmul(WeightType::LmHead, 0, y, vocab, d)?;
        Ok((one_hot, logits))
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
            let product = self.node(Op::Mul, &[a, up], virt(&[n, ff]))?;
            self.rescale(product, point(Site::Product), ff, INT8)?
        } else {
            self.node(Op::Mul, &[a, up], virt(&[n, ff]))?
        };
        let acc = self.matmul(WeightType::Down, layer, h, d, ff)?;
        self.residual(acc, residual, point(Site::MlpResidual))
    }

    /// The saturating sum `clamp(res + ⌊(acc + 2^{r−1}) / 2^r⌋)` of the residual and the
    /// accumulator at `point`, within `[−2^{bits−1}, 2^{bits−1} − 1]` (RFC-0022):
    ///
    /// - a site of one class, with `2^r` a constant, rescales the fused key `acc + 2^r · res`,
    ///   whose rounding is the accumulator's since `2^r · res` is a multiple of `2^r`;
    /// - otherwise the classes' shifts are hidden: `u = rq(acc, r_c)` within `±(2^{bits} − 1)`,
    ///   then `rq(2(res + u), 1)` clamps the sum. The first clamp binds only where the sum
    ///   saturates anyway.
    fn residual(
        &mut self,
        acc: TensorId,
        residual: TensorId,
        point: QuantPoint,
    ) -> Result<TensorId, CompileError> {
        let (n, d) = (self.n, self.model.arch.d_model);
        let bits = self.model.quant.residual_bits;
        let half = 1i64 << (bits - 1);
        let shifts = shifts(self.model.quant, point)?;
        if let [r] = shifts[..]
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
            return Ok(self.node(op, &[key], advice(&[n, d]))?);
        }
        let update = self.rescale(acc, point, d, (1 - 2 * half, 2 * half - 1))?;
        let sum = self.node(Op::Add, &[residual, update], virt(&[n, d]))?;
        let two = self.graph.constant(vec![2])?;
        let doubled = self.node(Op::ConstMul { c: two }, &[sum], virt(&[n, d]))?;
        let shift = self.graph.shift_classes(vec![1])?;
        let op = Op::Requant {
            shift,
            lo: -half,
            hi: half - 1,
        };
        Ok(self.node(op, &[doubled], advice(&[n, d]))?)
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
        let scaled = self.node(Op::Mul, &[residual, rho], virt(&[n, d]))?;
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

    /// `rq` at `point` on `[n, channels]`, saturating to `range`, with the channel classes of the
    /// weight before it when the site has several.
    fn rescale(
        &mut self,
        x: TensorId,
        point: QuantPoint,
        channels: u32,
        (lo, hi): (i64, i64),
    ) -> Result<TensorId, CompileError> {
        let shifts = shifts(self.model.quant, point)?;
        let count = shifts.len();
        let shift = self.graph.shift_classes(shifts)?;
        let mut inputs = vec![x];
        if count > 1 {
            inputs.push(self.class_vector(point, channels, count)?);
        }
        Ok(self.node(
            Op::Requant { shift, lo, hi },
            &inputs,
            advice(&[self.n, channels]),
        )?)
    }

    /// The committed classes of the output channels of the weight before `point`.
    fn class_vector(
        &mut self,
        point: QuantPoint,
        channels: u32,
        count: usize,
    ) -> Result<TensorId, CompileError> {
        let weight = match point {
            QuantPoint::Layer { layer, site } => after(site).map(|ty| WeightRef { ty, layer }),
            QuantPoint::Embed | QuantPoint::FinalNorm => None,
        }
        .ok_or(CompileError::Model {
            rule: "a site after no weight has one class",
        })?;
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
        let (queries, keys) = (self.queries, self.keys);
        let q = self.heads(q, h, queries)?;
        let k = self.heads(k, kv, keys)?;
        let v = self.heads(v, kv, keys)?;
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
        let exp = self.table(TableKind::Exp {
            key_bits: bits(spread),
            sigma: quant.exp_sigma,
            scale,
        })?;
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
        let o = self.node(Op::Mul, &[pv, r], virt(&[h, n, dh]))?;
        let rows = self.node(Op::Reshape, &[o], virt(&[h * n, dh]))?;
        let back = self.back;
        let rows = self.node(
            Op::GatherPublic { index: back },
            &[rows],
            virt(&[n * h, dh]),
        )?;
        Ok(self.node(Op::Reshape, &[rows], virt(&[n, h * dh]))?)
    }

    /// `[n, heads · d_head]` gathered into `[n_heads, n, d_head]` by `index`.
    fn heads(
        &mut self,
        x: TensorId,
        heads: u32,
        index: ConstRef,
    ) -> Result<TensorId, CompileError> {
        let arch = self.model.arch;
        let (n, h, dh) = (self.n, arch.n_heads, arch.d_head);
        let rows = self.node(Op::Reshape, &[x], virt(&[n * heads, dh]))?;
        let gathered = self.node(Op::GatherPublic { index }, &[rows], virt(&[h * n, dh]))?;
        Ok(self.node(Op::Reshape, &[gathered], virt(&[h, n, dh]))?)
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
        Ok(self.node(
            Op::TableMap { table },
            &inputs,
            decl(&[n, ff], NodeKind::LookupOutput { table }),
        )?)
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
