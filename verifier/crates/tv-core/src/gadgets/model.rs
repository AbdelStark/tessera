// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Model-owned native reductions and canonical fixed leaf bindings (RFC-0009/0037/0045/0050).
//! The result still owes canonical openings and the greedy certificate; it is never Accept.

use super::{GadgetError, Plan as Gadgets, greedy, root, runtime, softmax, stacked};
use crate::advice::Source;
use crate::leaf::{self, split};
use crate::lowering::{Graph, NodeId, Op, Role as GraphRole, TensorId};
use crate::schedule::{self, Native, NativeKind, NativeOutput, Schedule, ScheduleError};
use crate::stages::{
    self, Composed, LeafValues, NativeChecks, NativeContext, NativeFinal, StageError,
};
use crate::sumcheck::Instance;
use tv_field::{Fp, Fp2};
use tv_tables::TableId;

type Result<T> = core::result::Result<T, GadgetError>;
fn internal() -> GadgetError {
    GadgetError::Internal
}
fn reserve<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| GadgetError::Memory)?;
    Ok(v)
}
fn copy<T: Copy>(v: &[T]) -> Result<Vec<T>> {
    let mut out = reserve(v.len())?;
    out.extend_from_slice(v);
    Ok(out)
}
fn list<T, const N: usize>(items: [T; N]) -> Result<Vec<T>> {
    let mut out = reserve(N)?;
    out.extend(items);
    Ok(out)
}
fn push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    v.try_reserve(1).map_err(|_| GadgetError::Memory)?;
    v.push(value);
    Ok(())
}
fn claims_error(e: crate::validation::ClaimsError) -> GadgetError {
    match e {
        crate::validation::ClaimsError::Memory => GadgetError::Memory,
        _ => internal(),
    }
}
fn split_error(e: split::Error) -> GadgetError {
    match e {
        split::Error::Memory => GadgetError::Memory,
        _ => internal(),
    }
}
fn schedule_error(e: ScheduleError) -> GadgetError {
    match e {
        ScheduleError::Memory => GadgetError::Memory,
        _ => internal(),
    }
}
fn stage_error(e: GadgetError) -> StageError {
    match e {
        GadgetError::Memory => StageError::Memory,
        GadgetError::Rejected { gadget } => StageError::Gadget { gadget },
        _ => StageError::Internal,
    }
}
fn bits(n: u32) -> Result<usize> {
    usize::try_from(
        u32::BITS.saturating_sub(n.checked_sub(1).ok_or_else(internal)?.leading_zeros()),
    )
    .map_err(|_| internal())
}
fn variables(graph: &Graph, tensor: TensorId) -> Result<usize> {
    graph
        .tensor(tensor)
        .ok_or_else(internal)?
        .shape
        .dims()
        .iter()
        .try_fold(0usize, |a, &n| a.checked_add(bits(n)?).ok_or_else(internal))
}
/// The point at which the leaf stage left its claim on a tensor of `vars` variables with
/// `token` its token axis (RFC-0045 'The leaf point', RFC-0054 'The stage's outputs').
fn at(point: &split::LeafPoint, vars: usize, token: Option<split::TokenAxis>) -> Result<Vec<Fp2>> {
    point
        .tensor(u32::try_from(vars).map_err(|_| internal())?, token)
        .ok_or_else(internal)
}
fn own_point(p: &[Fp2], n: usize) -> Result<&[Fp2]> {
    p.get(p.len().checked_sub(n).ok_or_else(internal)?..)
        .ok_or_else(internal)
}

#[derive(Clone, Copy)]
enum Relation {
    Product(usize),
    RootPadding,
    ResidualPadding,
    Key,
    Forest,
    Reciprocal,
    MaximumPadding,
    ReciprocalPadding,
    Tokens,
    GreedyKey,
    GreedySelected,
    ObserveLogits,
}
impl Relation {
    fn kind(self) -> NativeKind {
        match self {
            Self::Product(_) => NativeKind::Product,
            Self::RootPadding => NativeKind::RootPad,
            Self::ResidualPadding => NativeKind::ResidualPad,
            Self::Key => NativeKind::Key,
            Self::Forest => NativeKind::Forest,
            Self::Reciprocal => NativeKind::Reciprocal,
            Self::MaximumPadding => NativeKind::MaxPad,
            Self::ReciprocalPadding => NativeKind::ReciprocalPad,
            Self::Tokens => NativeKind::Tokens,
            Self::GreedyKey => NativeKind::GreedyKey,
            Self::GreedySelected => NativeKind::GreedySelected,
            Self::ObserveLogits => NativeKind::Observe,
        }
    }
}
enum Family<'p> {
    Root(stacked::RootStack<'p>),
    Softmax {
        stack: stacked::SoftmaxStack<'p>,
        forest: Option<usize>,
    },
    Tokens {
        node: NodeId,
        indicator: TensorId,
        tokens: TensorId,
        nv: usize,
        nt: usize,
    },
    Greedy {
        node: NodeId,
        certificate: runtime::Certificate,
    },
    ObserveLogits {
        node: NodeId,
        tensor: TensorId,
        vars: usize,
    },
}
impl Family<'_> {
    fn layer_variables(&self) -> usize {
        match self {
            Self::Root(s) => s.layer_variables(),
            Self::Softmax { stack, .. } => stack.layer_variables(),
            Self::Tokens { .. } | Self::Greedy { .. } | Self::ObserveLogits { .. } => 0,
        }
    }
    fn node(&self) -> Result<NodeId> {
        match self {
            Self::Root(s) => s.sites().first().map(|s| s.node()),
            Self::Softmax { stack, .. } => stack.sites().first().map(|s| s.node()),
            Self::Tokens { node, .. }
            | Self::Greedy { node, .. }
            | Self::ObserveLogits { node, .. } => Some(*node),
        }
        .ok_or_else(internal)
    }
}
struct Projection {
    claim: usize,
    sources: Vec<Source>,
    from: usize,
    vars: usize,
}
impl Projection {
    fn new(
        claim: usize,
        sources: impl Iterator<Item = Source>,
        from: usize,
        vars: usize,
    ) -> Result<Self> {
        let mut actual = Vec::new();
        for s in sources {
            push(&mut actual, s)?;
        }
        Ok(Self {
            claim,
            sources: actual,
            from,
            vars,
        })
    }
    fn value(&self, claim: &stacked::Claim, point: &[Fp2], lambda: usize) -> Result<Fp2> {
        let local = point.len().checked_sub(lambda).ok_or_else(internal)?;
        let end = self.from.checked_add(self.vars).ok_or_else(internal)?;
        if end > local
            || claim.sources != self.sources
            || claim.point != point.get(self.from..end).ok_or_else(internal)?
            || claim.layer_point != point.get(local..).ok_or_else(internal)?
        {
            return Err(internal());
        }
        Ok(claim.value)
    }
    fn metadata(&self) -> Result<NativeOutput> {
        let Source::Tensor(tensor) = self.sources.first().copied().ok_or_else(internal)? else {
            return Err(internal());
        };
        Ok(NativeOutput {
            tensor,
            from: self.from,
            vars: self.vars,
        })
    }
}
struct Slot {
    family: usize,
    relation: Relation,
    outputs: Vec<Projection>,
}
fn append(
    slots: &mut Vec<Slot>,
    native: &mut Vec<Native>,
    slot: Slot,
    mut spec: Native,
) -> Result<()> {
    spec.tag =
        u32::try_from(slots.len().checked_add(1).ok_or_else(internal)?).map_err(|_| internal())?;
    spec.outputs = reserve(slot.outputs.len())?;
    for out in &slot.outputs {
        spec.outputs.push(out.metadata()?);
    }
    push(native, spec)?;
    push(slots, slot)
}

/// Complete current-model native slots, derived from the independent gadget plan.
/// Own source identities and coordinates are fixed before any proof value is read.
pub struct Plan<'p> {
    gadgets: &'p Gadgets<'p>,
    sites: Vec<split::Site>,
    layout: split::Layout,
    /// Every tensor's token axis, which places a leaf claim in a grouped stage (RFC-0054).
    axes: leaf::groups::TokenAxes,
    families: Vec<Family<'p>>,
    slots: Vec<Slot>,
    native: Vec<Native>,
    schedule: Schedule,
    leaf_height: usize,
}
impl<'p> Plan<'p> {
    /// Derive every root coefficient, masked key/forest, reciprocal and padding relation.
    /// No proof supplies descriptors or callback identities.
    ///
    /// # Errors
    /// Independent metadata errors become `Internal`, allocation failures `Memory`.
    pub fn derive(gadgets: &'p Gadgets<'p>) -> Result<Self> {
        Self::derive_inner(gadgets, None)
    }
    /// Compose all model checks with canonical token/greedy bindings from one runtime catalog.
    /// No external token/logit scalar substitutes for a source or a native relation.
    ///
    /// # Errors
    /// Own metadata errors `Internal`; all failed reservations `Memory`.
    pub fn derive_runtime(runtime: &'p runtime::Plan<'p>) -> Result<Self> {
        Self::derive_inner(runtime.gadgets(), Some(runtime))
    }
    fn derive_inner(gadgets: &'p Gadgets<'p>, runtime: Option<&runtime::Plan<'p>>) -> Result<Self> {
        let graph = gadgets.graph;
        let sites = match runtime {
            Some(r) => r.split_sites(),
            None => split::model_sites(graph, gadgets.policy),
        }
        .map_err(claims_error)?;
        let layout = split::Layout::new(&sites).map_err(split_error)?;
        let axes = leaf::groups::TokenAxes::derive(graph).map_err(|e| match e {
            leaf::groups::AxisError::Memory => GadgetError::Memory,
            _ => internal(),
        })?;
        let leaf = match runtime {
            Some(r) => r.leaf_schedule(),
            None => leaf::ModelSchedule::derive(graph, gadgets.policy),
        }
        .map_err(claims_error)?;
        let mut forest_trees = leaf
            .schedule()
            .trees()
            .iter()
            .enumerate()
            .filter_map(|(i, t)| matches!(t, leaf::Tree::Forest { .. }).then_some(i));
        let mut families = Vec::new();
        for site in &sites {
            let node = site.layers.first().ok_or_else(internal)?.node;
            match graph.node(node).ok_or_else(internal)?.op() {
                Op::RsqrtRow { .. } => {
                    push(&mut families, Family::Root(gadgets.root_stack(node)?))?
                }
                Op::SoftmaxRow { .. } => {
                    let forest = if graph.length() > 1 {
                        Some(forest_trees.next().ok_or_else(internal)?)
                    } else {
                        None
                    };
                    push(
                        &mut families,
                        Family::Softmax {
                            stack: gadgets.softmax_stack(node)?,
                            forest,
                        },
                    )?;
                }
                _ => {}
            }
        }
        if forest_trees.next().is_some() {
            return Err(internal());
        }
        let mut slots = Vec::new();
        let mut native = Vec::new();
        let mut deferred = Vec::new();
        for (f, family) in families.iter().enumerate() {
            let node = family.node()?;
            let lambda = family.layer_variables();
            let mut add = |relation,
                           vars,
                           degree,
                           evaluations,
                           zero,
                           outputs: Vec<Projection>|
             -> Result<()> {
                append(
                    &mut slots,
                    &mut native,
                    Slot {
                        family: f,
                        relation,
                        outputs,
                    },
                    Native {
                        kind: relation.kind(),
                        tag: 0,
                        node,
                        vars,
                        degree,
                        evaluations,
                        zero,
                        outputs: Vec::new(),
                    },
                )
            };
            match family {
                Family::Root(stack) => {
                    let root = stack.sites().first().ok_or_else(internal)?;
                    let shape = root.product_geometry();
                    let vars = root.padding_instance().vars;
                    let mut squares = reserve(stack.sites().len())?;
                    let mut rho = reserve(stack.sites().len())?;
                    for s in stack.sites() {
                        squares.push(*s.sources().first().ok_or_else(internal)?);
                        rho.push(*s.sources().get(1).ok_or_else(internal)?);
                    }
                    for j in 0..shape.coefficients().checked_mul(2).ok_or_else(internal)? {
                        add(
                            Relation::Product(j),
                            vars.checked_add(lambda).ok_or_else(internal)?,
                            4,
                            shape.evaluations(),
                            false,
                            list([
                                Projection::new(0, squares.iter().copied(), 0, vars)?,
                                Projection::new(1, rho.iter().copied(), 0, vars)?,
                            ])?,
                        )?;
                    }
                    add(
                        Relation::RootPadding,
                        vars.checked_add(lambda).ok_or_else(internal)?,
                        3,
                        1,
                        true,
                        list([Projection::new(0, rho.into_iter(), 0, vars)?])?,
                    )?;
                    let residual = root.residual_padding_instance().vars;
                    add(
                        Relation::ResidualPadding,
                        residual.checked_add(lambda).ok_or_else(internal)?,
                        3,
                        1,
                        true,
                        list([Projection::new(
                            0,
                            stack.sites().iter().map(root::Root::residual_source),
                            0,
                            residual,
                        )?])?,
                    )?;
                }
                Family::Softmax { stack, .. } => {
                    let site = *stack.sites().first().ok_or_else(internal)?;
                    let vars = site.variables();
                    let row = site.row_variables();
                    for relation in [Relation::Key, Relation::Forest] {
                        let mut sources = reserve(stack.sites().len())?;
                        for s in stack.sites() {
                            sources.push(Source::Tensor(
                                *graph
                                    .node(s.node())
                                    .and_then(|n| n.inputs().first())
                                    .ok_or_else(internal)?,
                            ));
                        }
                        add(
                            relation,
                            vars.checked_add(lambda).ok_or_else(internal)?,
                            3,
                            2,
                            false,
                            list([Projection::new(1, sources.into_iter(), 0, vars)?])?,
                        )?;
                    }
                    let mut reciprocal = reserve(stack.sites().len())?;
                    for s in stack.sites() {
                        reciprocal.push(s.source(0).ok_or_else(internal)?);
                    }
                    add(
                        Relation::Reciprocal,
                        row.checked_add(lambda).ok_or_else(internal)?,
                        3,
                        2,
                        false,
                        list([Projection::new(0, reciprocal.iter().copied(), 0, row)?])?,
                    )?;
                    add(
                        Relation::MaximumPadding,
                        row.checked_add(lambda).ok_or_else(internal)?,
                        3,
                        1,
                        true,
                        Vec::new(),
                    )?;
                    add(
                        Relation::ReciprocalPadding,
                        row.checked_add(lambda).ok_or_else(internal)?,
                        3,
                        1,
                        true,
                        list([Projection::new(0, reciprocal.into_iter(), 0, row)?])?,
                    )?;
                    push(&mut deferred, (node, Some(0)))?;
                }
                _ => return Err(internal()),
            }
        }
        if let Some(runtime) = runtime {
            let first = graph.nodes().first().ok_or_else(internal)?;
            let Op::OneHot { vocab } = *first.op() else {
                return Err(internal());
            };
            let node = NodeId(0);
            let indicator = *first.outputs().first().ok_or_else(internal)?;
            let tokens = *first.inputs().first().ok_or_else(internal)?;
            let nv = bits(vocab)?;
            let nt = bits(graph.length())?;
            let vars = nv.checked_add(nt).ok_or_else(internal)?;
            let family = families.len();
            push(
                &mut families,
                Family::Tokens {
                    node,
                    indicator,
                    tokens,
                    nv,
                    nt,
                },
            )?;
            append(
                &mut slots,
                &mut native,
                Slot {
                    family,
                    relation: Relation::Tokens,
                    outputs: list([
                        Projection::new(0, [Source::Tensor(indicator)].into_iter(), 0, vars)?,
                        Projection::new(1, [Source::Tensor(tokens)].into_iter(), nv, nt)?,
                    ])?,
                },
                Native {
                    kind: NativeKind::Tokens,
                    tag: 0,
                    node,
                    vars,
                    degree: 2,
                    evaluations: 2,
                    zero: true,
                    outputs: Vec::new(),
                },
            )?;
            push(&mut deferred, (node, Some(0)))?;
            let last = graph.nodes().last().ok_or_else(internal)?;
            let node = NodeId(
                u32::try_from(graph.nodes().len().checked_sub(1).ok_or_else(internal)?)
                    .map_err(|_| internal())?,
            );
            let logits = *last.outputs().first().ok_or_else(internal)?;
            let vars = variables(graph, logits)?;
            let family = families.len();
            if let Some(certificate) = runtime.certificate() {
                push(&mut families, Family::Greedy { node, certificate })?;
                for (relation, kind, zero) in [
                    (Relation::GreedyKey, greedy::Kind::Key, false),
                    (Relation::GreedySelected, greedy::Kind::Selected, true),
                ] {
                    let boundary = certificate.boundary();
                    append(
                        &mut slots,
                        &mut native,
                        Slot {
                            family,
                            relation,
                            outputs: list([Projection::new(
                                0,
                                [Source::Tensor(logits)].into_iter(),
                                0,
                                vars,
                            )?])?,
                        },
                        Native {
                            kind: relation.kind(),
                            tag: 0,
                            node,
                            vars,
                            degree: boundary.degree(kind),
                            evaluations: boundary.evaluations(kind),
                            zero,
                            outputs: Vec::new(),
                        },
                    )?;
                }
            } else {
                push(
                    &mut families,
                    Family::ObserveLogits {
                        node,
                        tensor: logits,
                        vars,
                    },
                )?;
                append(
                    &mut slots,
                    &mut native,
                    Slot {
                        family,
                        relation: Relation::ObserveLogits,
                        outputs: list([Projection::new(
                            0,
                            [Source::Tensor(logits)].into_iter(),
                            0,
                            vars,
                        )?])?,
                    },
                    Native {
                        kind: NativeKind::Observe,
                        tag: 0,
                        node,
                        vars,
                        degree: 1,
                        evaluations: 1,
                        zero: true,
                        outputs: Vec::new(),
                    },
                )?;
            }
            push(&mut deferred, (node, None))?;
        }
        let schedule =
            schedule::derive_composed(graph, &deferred, &native).map_err(schedule_error)?;
        Ok(Self {
            gadgets,
            sites,
            layout,
            axes,
            families,
            slots,
            native,
            schedule,
            leaf_height: usize::try_from(leaf.schedule().height()).map_err(|_| internal())?,
        })
    }
    /// Own complete mixed graph/native schedule.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }
    /// Own native metadata, in deterministic tag order, for diagnostics and DEP-2 comparison.
    pub fn native(&self) -> &[Native] {
        &self.native
    }
}

/// Token and logit evaluations whose anchor and greedy bindings remain outstanding.
/// The model stages only reduce these evaluations to original commitments.
#[derive(Clone, Copy)]
pub struct External<'a> {
    /// OneHot's original token-stream evaluation.
    pub tokens: &'a LeafValues,
    /// Original model logits at the certificate's prescribed point.
    pub logits: &'a LeafValues,
}
struct Data {
    point: Vec<Fp2>,
    score_point: Vec<Fp2>,
    sources: Vec<Fp2>,
    keys: Vec<Fp2>,
    values: Vec<Fp2>,
}
/// Fixed canonical model bindings, checked before the mixed stage's gamma.
/// [`Prepared::verify`] checks the mixed stages and retains every original source debt.
#[must_use = "verify the complete mixed stages and bind every returned original source"]
pub struct Prepared<'a, 'p> {
    plan: &'a Plan<'p>,
    bound: &'a split::Bound,
    data: Vec<Option<Data>>,
    columns: Vec<Option<(usize, usize)>>,
    leaf: Vec<LeafValues>,
    original: Vec<stacked::Claim>,
    external: Option<External<'a>>,
}
/// Complete model reductions, with all remaining canonical opening and external debts.
#[must_use = "check graph/source/multiplicity openings and the token/greedy certificate bindings"]
pub struct Reduced<'a> {
    /// Every graph reduction's original commitment pool.
    pub graph: stages::Verified,
    /// Original auxiliary and committed tensor claims, including reciprocal half-key points.
    pub sources: Vec<stacked::Claim>,
    /// Multiplicities retain their table identities and original table-tree points.
    pub multiplicities: &'a [(TableId, crate::logup::Claim)],
    /// Bind these same original token/logit claims to the anchor/greedy certificate.
    pub external: Option<External<'a>>,
}
fn own_claims<'a>(claims: impl Iterator<Item = super::Claim<'a>>) -> Result<Vec<stacked::Claim>> {
    let mut out = Vec::new();
    for c in claims {
        push(
            &mut out,
            stacked::Claim {
                sources: copy(&[c.input.source])?,
                point: copy(c.input.point)?,
                layer_point: Vec::new(),
                value: c.value,
            },
        )?;
    }
    Ok(out)
}
fn input_source(graph: &Graph, node: NodeId, source: Source) -> Result<Option<usize>> {
    let Source::Tensor(t) = source else {
        return Ok(None);
    };
    Ok(graph
        .node(node)
        .ok_or_else(internal)?
        .inputs()
        .iter()
        .position(|&i| i == t))
}
impl<'p> Plan<'p> {
    /// Bind the exact canonical leaf layout and check all fixed gadget constraints.
    /// `bound` must be the output of the production canonical split checker after shared trees.
    /// External claims remain explicitly owed to anchor/scoring, never inferred from this check.
    ///
    /// # Errors
    /// Wrong own identities/points/counts give `Internal`; gadget equations retain their exact
    /// `Rejected { gadget }`; every failed reserve is `Memory`.
    pub fn bind<'a>(
        &'a self,
        bound: &'a split::Bound,
        external: External<'a>,
    ) -> Result<Prepared<'a, 'p>> {
        self.bind_inner(bound, Some(external))
    }
    /// Bind runtime leaves without an external token/logit scalar. All such sources must
    /// have the exact native replacement in the derived schedule.
    ///
    /// # Errors
    /// Same fixed-gadget errors as bind; absent own replacement metadata is Internal.
    pub fn bind_runtime<'a>(&'a self, bound: &'a split::Bound) -> Result<Prepared<'a, 'p>> {
        self.bind_inner(bound, None)
    }
    fn bind_inner<'a>(
        &'a self,
        bound: &'a split::Bound,
        external: Option<External<'a>>,
    ) -> Result<Prepared<'a, 'p>> {
        self.layout.check_bound(bound).map_err(split_error)?;
        // An ungrouped stage's `P` is its tallest tree's (RFC-0045 'The leaf point'). A grouped
        // stage's merge has its own `n_max` rounds, which every merged claim's point fits
        // (RFC-0054 'The merge's rounds'): `LeafPoint::tensor` has no point beyond them.
        if let split::LeafPoint::Shared(p) = &bound.point
            && p.len() != self.leaf_height
        {
            return Err(internal());
        }
        if bound.multiplicities.len() != self.layout.instances().count() {
            return Err(internal());
        }
        for ((id, claim), (expected, _)) in bound.multiplicities.iter().zip(self.layout.instances())
        {
            let table = self
                .gadgets
                .selected
                .tables
                .get(usize::try_from(id.0).map_err(|_| internal())?)
                .ok_or_else(internal)?;
            if *id != expected || claim.point.len() != usize::from(table.log2_rows) {
                return Err(internal());
            }
        }
        let mut forests = bound.forests.iter();
        for (i, family) in self.families.iter().enumerate() {
            if let Family::Softmax {
                forest: Some(tree), ..
            } = family
            {
                let f = forests.next().ok_or_else(internal)?;
                let vars = self
                    .slots
                    .iter()
                    .enumerate()
                    .find_map(|(j, s)| {
                        (s.family == i && matches!(s.relation, Relation::Forest)).then_some(j)
                    })
                    .and_then(|j| self.native.get(j))
                    .ok_or_else(internal)?
                    .vars;
                if f.tree != *tree || f.claim.point.len() != vars {
                    return Err(internal());
                }
            }
        }
        if forests.next().is_some() {
            return Err(internal());
        }
        let graph = self.gadgets.graph;
        let mut columns = reserve(graph.nodes().len())?;
        columns.resize(graph.nodes().len(), None);
        for (i, column) in bound.columns.iter().enumerate() {
            let slot = columns
                .get_mut(usize::try_from(column.node.0).map_err(|_| internal())?)
                .ok_or_else(internal)?;
            let end = i.checked_add(1).ok_or_else(internal)?;
            *slot = Some(match *slot {
                None => (i, end),
                Some((start, previous)) if previous == i => (start, end),
                _ => return Err(internal()),
            });
        }
        let mut data = reserve(graph.nodes().len())?;
        data.resize_with(graph.nodes().len(), || None);
        let mut prepared = Prepared {
            plan: self,
            bound,
            data,
            columns,
            leaf: Vec::new(),
            original: Vec::new(),
            external,
        };
        for site in &self.sites {
            for layer in &site.layers {
                let definition = graph.node(layer.node).ok_or_else(internal)?;
                let first = layer.lookups.first().ok_or_else(internal)?;
                // The site's point `r_s`: its blocks' (RFC-0045 'A site's point'); for Softmax its
                // row tensors', whose blocks are its requests after the Exp lookup (RFC-0054
                // 'Rows').
                let (vars, token) = if matches!(definition.op(), Op::SoftmaxRow { .. }) {
                    (
                        variables(graph, *definition.outputs().get(1).ok_or_else(internal)?)?,
                        layer.lookups.get(1).ok_or_else(internal)?.token,
                    )
                } else {
                    (
                        usize::try_from(first.variables).map_err(|_| internal())?,
                        first.token,
                    )
                };
                let mut keys = reserve(layer.lookups.len())?;
                let mut values = reserve(layer.lookups.len())?;
                for lookup in &layer.lookups {
                    keys.push(prepared.affine(layer.node, &lookup.key, lookup.token)?);
                    values.push(match &lookup.value {
                        Some(v) => prepared.affine(layer.node, v, lookup.token)?,
                        None => Fp2::ZERO,
                    });
                }
                let score_vars = usize::try_from(first.variables).map_err(|_| internal())?;
                *prepared
                    .data
                    .get_mut(usize::try_from(layer.node.0).map_err(|_| internal())?)
                    .ok_or_else(internal)? = Some(Data {
                    point: at(&bound.point, vars, token)?,
                    score_point: at(&bound.point, score_vars, first.token)?,
                    sources: Vec::new(),
                    keys,
                    values,
                });
            }
        }
        prepared.check_fixed()?;
        prepared.bind_leaf()?;
        Ok(prepared)
    }
}
impl<'a, 'p> Prepared<'a, 'p> {
    fn row(&self, node: NodeId) -> Result<&Data> {
        self.data
            .get(usize::try_from(node.0).map_err(|_| internal())?)
            .and_then(Option::as_ref)
            .ok_or_else(internal)
    }
    fn row_mut(&mut self, node: NodeId) -> Result<&mut Data> {
        self.data
            .get_mut(usize::try_from(node.0).map_err(|_| internal())?)
            .and_then(Option::as_mut)
            .ok_or_else(internal)
    }
    fn column(&self, node: NodeId, column: split::Column) -> Result<&split::ColumnClaim> {
        let (start, end) = self
            .columns
            .get(usize::try_from(node.0).map_err(|_| internal())?)
            .copied()
            .flatten()
            .ok_or_else(internal)?;
        self.bound
            .columns
            .get(start..end)
            .ok_or_else(internal)?
            .iter()
            .find(|c| c.column == column)
            .ok_or_else(internal)
    }
    /// A lookup form's value from its terms' split claims, each at its read's point of a block
    /// whose token axis is `token`.
    fn affine(
        &self,
        node: NodeId,
        form: &split::Affine,
        token: Option<split::TokenAxis>,
    ) -> Result<Fp2> {
        form.terms
            .iter()
            .try_fold(Fp2::from(form.constant), |a, (coefficient, read)| {
                let column = self.column(node, read.column)?;
                if self.bound.point.read(&read.coordinates, token).as_ref() != Some(&column.point) {
                    return Err(internal());
                }
                Ok(a + Fp2::from(*coefficient) * column.value)
            })
    }
    /// The point at which the leaf stage left its claim on `tensor`, by the tensor's own
    /// variables and token axis (RFC-0054 'Where a claim lies').
    fn tensor_point(&self, tensor: TensorId) -> Result<Vec<Fp2>> {
        let graph = self.plan.gadgets.graph;
        let token = split::token_axis(graph, &self.plan.axes, tensor).map_err(|_| internal())?;
        at(&self.bound.point, variables(graph, tensor)?, token)
    }
    fn source(&self, node: NodeId, source: Source, point: &[Fp2]) -> Result<Fp2> {
        let graph = self.plan.gadgets.graph;
        let definition = graph.node(node).ok_or_else(internal)?;
        let column = match source {
            Source::Auxiliary(owner, aux) if owner == node => split::Column::Auxiliary(aux),
            Source::Tensor(t) => {
                if let Some(i) = definition.inputs().iter().position(|&i| i == t) {
                    if i == 0
                        && let Op::RsqrtRow { eps_int, .. } = definition.op()
                    {
                        let c = self.column(node, split::Column::RootInput)?;
                        if c.point != point {
                            return Err(internal());
                        }
                        let psi = crate::selectors::lt(point, u64::from(graph.length()))
                            .ok_or_else(internal)?;
                        return Ok(c.value - Fp2::from(Fp::reduce(*eps_int)) * psi);
                    }
                    split::Column::Input(u8::try_from(i).map_err(|_| internal())?)
                } else if let Some(i) = definition.outputs().iter().position(|&i| i == t) {
                    split::Column::Output(u8::try_from(i).map_err(|_| internal())?)
                } else {
                    return Err(internal());
                }
            }
            _ => return Err(internal()),
        };
        let c = self.column(node, column)?;
        if c.point != point {
            return Err(internal());
        }
        Ok(c.value)
    }
    fn retain(&mut self, claims: Vec<stacked::Claim>) -> Result<()> {
        self.original
            .try_reserve(claims.len())
            .map_err(|_| GadgetError::Memory)?;
        self.original.extend(claims);
        Ok(())
    }
    fn check_fixed(&mut self) -> Result<()> {
        let graph = self.plan.gadgets.graph;
        for site in self.plan.gadgets.requantizations() {
            let site = site?;
            let node = site.node();
            let point = copy(&self.row(node)?.point)?;
            let mut sources = Vec::new();
            for input in site.inputs(&point)? {
                // A graph tensor's claim lies at its own point: a channel's class, read over the
                // channels alone, has no token axis (RFC-0054 'Where a claim lies'). In an
                // ungrouped stage that is the prefix `inputs` gives.
                let own = match input.source {
                    Source::Tensor(tensor) => self.tensor_point(tensor)?,
                    _ => copy(input.point)?,
                };
                push(&mut sources, self.source(node, input.source, &own)?)?;
            }
            let d = self.row(node)?;
            let mut lookups = reserve(d.keys.len())?;
            lookups.extend(d.keys.iter().copied().zip(d.values.iter().copied()));
            let checked = site.check(&point, &sources, &lookups)?;
            let mut kept = Vec::new();
            for c in checked.claims() {
                if input_source(graph, node, c.input.source)?.is_none() {
                    push(&mut kept, c)?;
                }
            }
            let claims = own_claims(kept.into_iter())?;
            self.retain(claims)?;
            self.row_mut(node)?.sources = sources;
        }
        for family in &self.plan.families {
            match family {
                Family::Tokens { .. } | Family::Greedy { .. } | Family::ObserveLogits { .. } => {}
                Family::Root(stack) => {
                    for site in stack.sites() {
                        let node = site.node();
                        let point = copy(&self.row(node)?.point)?;
                        let mut sources = reserve(site.sources().len())?;
                        for &s in site.sources() {
                            sources.push(self.source(node, s, &point)?);
                        }
                        let d = self.row(node)?;
                        let checked = site.check(&point, &sources, &d.keys, &d.values)?;
                        let claims = own_claims(checked.claims().skip(1))?;
                        self.retain(claims)?;
                        self.row_mut(node)?.sources = sources;
                    }
                }
                Family::Softmax { stack, .. } => {
                    for &site in stack.sites() {
                        let node = site.node();
                        let point = copy(&self.row(node)?.point)?;
                        let psi = Fp2::ONE - site.forest_root(&point)?;
                        let a = self.column(node, split::Column::SoftmaxA)?;
                        let b = self.column(node, split::Column::SoftmaxB)?;
                        if a.point != point || b.point != point {
                            return Err(internal());
                        }
                        let scale = Fp2::from(Fp::reduce(
                            1u64.checked_shl(
                                u32::try_from(bits(graph.length())?).map_err(|_| internal())?,
                            )
                            .ok_or_else(internal)?,
                        ));
                        let ehalf =
                            (a.value + b.value + psi) * scale.inverse().ok_or_else(internal)?;
                        let mut sources = Vec::new();
                        let mut index = 0usize;
                        while let Some(s) = site.source(index) {
                            push(
                                &mut sources,
                                if index == 1 {
                                    ehalf
                                } else {
                                    self.source(node, s, &point)?
                                },
                            )?;
                            index = index.checked_add(1).ok_or_else(internal)?;
                        }
                        let d = self.row(node)?;
                        let checked = site.check(
                            &point,
                            &sources,
                            d.keys.get(1..).ok_or_else(internal)?,
                            d.values.get(1..).ok_or_else(internal)?,
                        )?;
                        let claims = own_claims(checked.claims())?;
                        let e = self.column(node, split::Column::Output(0))?;
                        let exponential = stacked::Claim {
                            sources: copy(&[Source::Tensor(
                                *graph
                                    .node(node)
                                    .and_then(|n| n.outputs().first())
                                    .ok_or_else(internal)?,
                            )])?,
                            point: copy(&e.point)?,
                            layer_point: Vec::new(),
                            value: e.value,
                        };
                        self.retain(claims)?;
                        push(&mut self.original, exponential)?;
                        self.row_mut(node)?.sources = sources;
                    }
                }
            }
        }
        Ok(())
    }
    fn bind_leaf(&mut self) -> Result<()> {
        let graph = self.plan.gadgets.graph;
        for leaf in self.plan.schedule.leaf() {
            if leaf.input.is_none() {
                self.leaf.try_reserve(1).map_err(|_| GadgetError::Memory)?;
                self.leaf.push(copy_leaf(
                    self.external.ok_or_else(internal)?.logits,
                    leaf.claim.vars,
                )?);
                continue;
            }
            let node = graph.node(leaf.site).ok_or_else(internal)?;
            if matches!(node.op(), Op::OneHot { .. }) {
                self.leaf.try_reserve(1).map_err(|_| GadgetError::Memory)?;
                self.leaf.push(copy_leaf(
                    self.external.ok_or_else(internal)?.tokens,
                    leaf.claim.vars,
                )?);
                continue;
            }
            let input = leaf.input.ok_or_else(internal)?;
            let site = self
                .plan
                .sites
                .iter()
                .find(|s| s.layers.first().is_some_and(|l| l.node == leaf.site))
                .ok_or_else(internal)?;
            let mut values = reserve(site.layers.len())?;
            // The graph's claim on the gadget input, at the leaf stage's point for that tensor
            // (RFC-0037 'The layer point'; RFC-0054 'Where a claim lies'): every layer's tensor
            // of a template position has the same shape and token axis.
            let first = graph
                .node(site.layers.first().ok_or_else(internal)?.node)
                .ok_or_else(internal)?;
            let point = self.tensor_point(*first.inputs().get(input).ok_or_else(internal)?)?;
            if point.len() != leaf.claim.vars {
                return Err(internal());
            }
            let point = point.as_slice();
            for layer in &site.layers {
                let definition = graph.node(layer.node).ok_or_else(internal)?;
                let tensor = *definition.inputs().get(input).ok_or_else(internal)?;
                let class_zero = graph
                    .inputs()
                    .iter()
                    .any(|i| i.tensor == tensor && matches!(i.role, GraphRole::Classes(_)))
                    && self
                        .plan
                        .gadgets
                        .selected
                        .ranges
                        .get(usize::try_from(tensor.0).map_err(|_| internal())?)
                        .is_some_and(|r| r.bounds() == Some((0, 0)));
                values.push(if class_zero {
                    Fp2::ZERO
                } else {
                    self.source(layer.node, Source::Tensor(tensor), point)?
                });
            }
            if values.len()
                != if leaf.stacked {
                    usize::try_from(self.plan.schedule.layers()).map_err(|_| internal())?
                } else {
                    1
                }
            {
                return Err(internal());
            }
            push(
                &mut self.leaf,
                LeafValues {
                    point: copy(point)?,
                    values,
                },
            )?;
        }
        Ok(())
    }
    /// Graph leaf claims reconstructed from the exact canonical sources and external debts.
    pub fn leaf(&self) -> &[LeafValues] {
        &self.leaf
    }
}
fn copy_leaf(leaf: &LeafValues, vars: usize) -> Result<LeafValues> {
    if leaf.point.len() != vars || leaf.values.len() != 1 {
        return Err(internal());
    }
    Ok(LeafValues {
        point: copy(&leaf.point)?,
        values: copy(&leaf.values)?,
    })
}

impl Prepared<'_, '_> {
    fn slot(&self, tag: u32) -> Result<(&Slot, &Native)> {
        let index =
            usize::try_from(tag.checked_sub(1).ok_or_else(internal)?).map_err(|_| internal())?;
        let own = self.plan.native.get(index).ok_or_else(internal)?;
        if own.tag != tag {
            return Err(internal());
        }
        Ok((self.plan.slots.get(index).ok_or_else(internal)?, own))
    }
    fn context(&self, ctx: NativeContext<'_>) -> Result<()> {
        if ctx.stage != 1
            || ctx.layer.len() != self.plan.schedule.layer_vars()
            || ctx.zero.len()
                != self
                    .plan
                    .schedule
                    .stages()
                    .first()
                    .ok_or_else(internal)?
                    .zero_vars
        {
            return Err(internal());
        }
        Ok(())
    }
    fn layer<'c>(&self, family: &Family<'_>, point: &'c [Fp2]) -> Result<&'c [Fp2]> {
        let vars = family.layer_variables();
        if vars == 0 {
            Ok(&[])
        } else if point.len() == vars {
            Ok(point)
        } else {
            Err(internal())
        }
    }
    fn root<'c>(
        &'c self,
        stack: &'c stacked::RootStack<'_>,
        layer: &'c [Fp2],
    ) -> Result<stacked::RootChecked<'c, 'c>> {
        let mut input = reserve(stack.sites().len())?;
        for site in stack.sites() {
            let d = self.row(site.node())?;
            input.push(stacked::Layer {
                sources: &d.sources,
                keys: &d.keys,
                values: &d.values,
            });
        }
        let node = stack.sites().first().ok_or_else(internal)?.node();
        stack.check(&self.row(node)?.point, layer, &input)
    }
    fn reciprocal<'c>(
        &'c self,
        stack: &'c stacked::SoftmaxStack<'_>,
        layer: &'c [Fp2],
    ) -> Result<stacked::SoftmaxChecked<'c, 'c>> {
        let mut input = reserve(stack.sites().len())?;
        for site in stack.sites() {
            let d = self.row(site.node())?;
            input.push(stacked::Layer {
                sources: &d.sources,
                keys: d.keys.get(1..).ok_or_else(internal)?,
                values: d.values.get(1..).ok_or_else(internal)?,
            });
        }
        let node = stack.sites().first().ok_or_else(internal)?.node();
        stack.check(&self.row(node)?.point, layer, &input)
    }
    fn key<T>(
        &self,
        slot: &Slot,
        ctx: NativeContext<'_>,
        call: impl FnOnce(&stacked::Key<'_, '_>) -> Result<T>,
    ) -> Result<T> {
        let family = self.plan.families.get(slot.family).ok_or_else(internal)?;
        let Family::Softmax { stack, forest } = family else {
            return Err(internal());
        };
        let layer = self.layer(family, ctx.layer)?;
        let node = family.node()?;
        let data = self.row(node)?;
        match slot.relation {
            Relation::Key => {
                let mut fixed = reserve(stack.sites().len())?;
                for site in stack.sites() {
                    let d = self.row(site.node())?;
                    fixed.push((
                        *d.keys.first().ok_or_else(internal)?,
                        *d.values.first().ok_or_else(internal)?,
                    ));
                }
                call(&stack.lookup(&data.score_point, layer, &fixed)?)
            }
            Relation::Forest => {
                if let Some(tree) = forest {
                    let claim = self
                        .bound
                        .forests
                        .iter()
                        .find(|c| c.tree == *tree)
                        .ok_or_else(internal)?;
                    call(&stack.forest(&claim.claim.point, claim.claim.value)?)
                } else {
                    let site = *stack.sites().first().ok_or_else(internal)?;
                    let mut point = copy(&data.score_point)?;
                    point
                        .try_reserve(layer.len())
                        .map_err(|_| GadgetError::Memory)?;
                    point.extend_from_slice(layer);
                    let real = crate::selectors::lt(
                        layer,
                        u64::try_from(stack.sites().len()).map_err(|_| internal())?,
                    )
                    .ok_or_else(internal)?;
                    let root = Fp2::ONE - (Fp2::ONE - site.forest_root(&data.score_point)?) * real;
                    call(&stack.forest(&point, root)?)
                }
            }
            _ => Err(internal()),
        }
    }
    fn greedy<T>(
        &self,
        slot: &Slot,
        ctx: NativeContext<'_>,
        call: impl FnOnce(&greedy::Check<'_>) -> Result<T>,
    ) -> Result<T> {
        let Family::Greedy { node, certificate } =
            self.plan.families.get(slot.family).ok_or_else(internal)?
        else {
            return Err(internal());
        };
        let boundary = certificate.boundary();
        match slot.relation {
            Relation::GreedyKey => {
                let claim = self.column(*node, split::Column::GreedyKey)?;
                call(&boundary.key(&claim.point, claim.value)?)
            }
            Relation::GreedySelected => call(&boundary.zero(
                greedy::Kind::Selected,
                own_point(ctx.zero, bits(self.plan.gadgets.graph.length())?)?,
            )?),
            _ => Err(internal()),
        }
    }
    fn instance(&self, tag: u32, ctx: NativeContext<'_>) -> Result<Instance> {
        self.context(ctx)?;
        let (slot, meta) = self.slot(tag)?;
        let family = self.plan.families.get(slot.family).ok_or_else(internal)?;
        let layer = self.layer(family, ctx.layer)?;
        match (family, slot.relation) {
            (Family::Tokens { .. }, Relation::Tokens)
            | (Family::ObserveLogits { .. }, Relation::ObserveLogits) => Ok(Instance {
                claim: Fp2::ZERO,
                vars: meta.vars,
                evals: meta.evaluations,
            }),
            (Family::Greedy { .. }, Relation::GreedyKey | Relation::GreedySelected) => {
                self.greedy(slot, ctx, |c| Ok(c.instance()))
            }
            (Family::Root(stack), Relation::Product(i)) => self
                .root(stack, layer)?
                .products()
                .nth(i)
                .ok_or_else(internal)?
                .instance(),
            (Family::Root(stack), Relation::RootPadding) => stack.padding_instance(false),
            (Family::Root(stack), Relation::ResidualPadding) => stack.padding_instance(true),
            (Family::Softmax { .. }, Relation::Key | Relation::Forest) => {
                self.key(slot, ctx, |k| k.instance())
            }
            (Family::Softmax { stack, .. }, Relation::Reciprocal) => {
                self.reciprocal(stack, layer)?.instance()
            }
            (
                Family::Softmax { stack, .. },
                Relation::MaximumPadding | Relation::ReciprocalPadding,
            ) => stack.padding_instance(),
            _ => Err(internal()),
        }
    }
    fn component(
        &self,
        tag: u32,
        ctx: NativeContext<'_>,
        point: &[Fp2],
        values: &[Fp2],
    ) -> Result<stacked::Final> {
        self.context(ctx)?;
        let (slot, meta) = self.slot(tag)?;
        let family = self.plan.families.get(slot.family).ok_or_else(internal)?;
        let layer = self.layer(family, ctx.layer)?;
        match (family, slot.relation) {
            (Family::Greedy { .. }, Relation::GreedyKey | Relation::GreedySelected) => {
                let checked = self.greedy(slot, ctx, |c| c.evaluate(point, values))?;
                let mut claims = reserve(checked.claims.len())?;
                for c in checked.claims {
                    let source = match c.source {
                        greedy::Source::Logits(t)
                        | greedy::Source::OneHot(t)
                        | greedy::Source::Tokens(t) => Source::Tensor(t),
                        greedy::Source::Maximum(g) => Source::GreedyMaximum(g),
                    };
                    claims.push(stacked::Claim {
                        sources: list([source])?,
                        point: c.point,
                        layer_point: Vec::new(),
                        value: c.value,
                    });
                }
                Ok(stacked::Final {
                    value: checked.value,
                    claims,
                })
            }
            (
                Family::Tokens {
                    indicator,
                    tokens,
                    nv,
                    nt,
                    ..
                },
                Relation::Tokens,
            ) => {
                if point.len() != meta.vars {
                    return Err(internal());
                }
                if values.len() != 2 {
                    return Err(GadgetError::Rejected {
                        gadget: crate::validation::GadgetId(0),
                    });
                }
                let v = point.get(..*nv).ok_or_else(internal)?;
                let t = point.get(*nv..).ok_or_else(internal)?;
                let mut index = Fp2::ZERO;
                for (j, &r) in v.iter().enumerate() {
                    index += r * Fp2::from(Fp::reduce(
                        1u64.checked_shl(u32::try_from(j).map_err(|_| internal())?)
                            .ok_or_else(internal)?,
                    ));
                }
                let h = *values.first().ok_or_else(internal)?;
                let token = *values.get(1).ok_or_else(internal)?;
                let delta = v.iter().fold(Fp2::ONE, |a, &r| a * (Fp2::ONE - r));
                let value = crate::sumcheck::eq(own_point(ctx.zero, *nt)?, t)
                    .ok_or_else(internal)?
                    * (index * h - delta * token);
                Ok(stacked::Final {
                    value,
                    claims: list([
                        stacked::Claim {
                            sources: list([Source::Tensor(*indicator)])?,
                            point: copy(point)?,
                            layer_point: Vec::new(),
                            value: h,
                        },
                        stacked::Claim {
                            sources: list([Source::Tensor(*tokens)])?,
                            point: copy(t)?,
                            layer_point: Vec::new(),
                            value: token,
                        },
                    ])?,
                })
            }
            (Family::ObserveLogits { tensor, vars, .. }, Relation::ObserveLogits) => {
                if point.len() != *vars || values.len() != 1 {
                    return Err(internal());
                }
                Ok(stacked::Final {
                    value: Fp2::ZERO,
                    claims: list([stacked::Claim {
                        sources: list([Source::Tensor(*tensor)])?,
                        point: copy(point)?,
                        layer_point: Vec::new(),
                        value: *values.first().ok_or_else(internal)?,
                    }])?,
                })
            }
            (Family::Root(stack), Relation::Product(i)) => self
                .root(stack, layer)?
                .products()
                .nth(i)
                .ok_or_else(internal)?
                .evaluate(point, values),
            (Family::Root(stack), Relation::RootPadding) => {
                stack.padding(false, own_point(ctx.zero, meta.vars)?, point, values)
            }
            (Family::Root(stack), Relation::ResidualPadding) => {
                stack.padding(true, own_point(ctx.zero, meta.vars)?, point, values)
            }
            (Family::Softmax { .. }, Relation::Key | Relation::Forest) => {
                self.key(slot, ctx, |k| k.evaluate(point, values))
            }
            (Family::Softmax { stack, .. }, Relation::Reciprocal) => {
                self.reciprocal(stack, layer)?.evaluate(point, values)
            }
            (
                Family::Softmax { stack, .. },
                Relation::MaximumPadding | Relation::ReciprocalPadding,
            ) => stack.padding(
                if matches!(slot.relation, Relation::MaximumPadding) {
                    softmax::Padding::Maximum
                } else {
                    softmax::Padding::Reciprocal
                },
                own_point(ctx.zero, meta.vars)?,
                point,
                values,
            ),
            _ => Err(internal()),
        }
    }
    fn transform(
        &self,
        tag: u32,
        point: &[Fp2],
        checked: stacked::Final,
    ) -> Result<(NativeFinal, Vec<stacked::Claim>)> {
        let (slot, _) = self.slot(tag)?;
        let lambda = self
            .plan
            .families
            .get(slot.family)
            .ok_or_else(internal)?
            .layer_variables();
        let mut graph = reserve(slot.outputs.len())?;
        for out in &slot.outputs {
            let claim = checked.claims.get(out.claim).ok_or_else(internal)?;
            graph.push(out.value(claim, point, lambda)?);
        }
        let mut remaining = Vec::new();
        for (i, claim) in checked.claims.into_iter().enumerate() {
            if slot.outputs.iter().any(|o| o.claim == i) {
                continue;
            }
            for &source in &claim.sources {
                if let Source::Tensor(t) = source
                    && self.plan.gadgets.graph.tensor(t).is_none_or(|t| {
                        !matches!(
                            t.kind,
                            crate::lowering::Kind::Advice
                                | crate::lowering::Kind::Lookup { .. }
                                | crate::lowering::Kind::Weight
                        )
                    })
                {
                    return Err(internal());
                }
            }
            push(&mut remaining, claim)?;
        }
        Ok((
            NativeFinal {
                value: checked.value,
                graph,
            },
            remaining,
        ))
    }
}
impl NativeChecks for Prepared<'_, '_> {
    fn instance(
        &self,
        tag: u32,
        ctx: NativeContext<'_>,
    ) -> core::result::Result<Instance, StageError> {
        Prepared::instance(self, tag, ctx).map_err(stage_error)
    }
    fn evaluate(
        &self,
        tag: u32,
        ctx: NativeContext<'_>,
        point: &[Fp2],
        values: &[Fp2],
    ) -> core::result::Result<NativeFinal, StageError> {
        let result = self
            .component(tag, ctx, point, values)
            .and_then(|c| self.transform(tag, point, c).map(|(value, _)| value));
        result.map_err(stage_error)
    }
}
impl<'a> Prepared<'a, '_> {
    /// Check the ONE mixed stage loop without absorbing the canonical split twice. Every
    /// non-graph native source is reconstructed from its same checked raw final evaluations.
    ///
    /// # Errors
    /// Exact existing stage rejections; own postcheck geometry `Internal`, failed reserve `Memory`.
    pub fn verify(
        &self,
        transcript: &mut tv_hash::transcript::Transcript,
        rope: &tv_tables::rope::Rope,
        proofs: &[stages::StageProof],
    ) -> core::result::Result<Reduced<'a>, StageError> {
        let checked = stages::verify_composed_bound(
            transcript,
            &self.plan.schedule,
            rope,
            &self.leaf,
            self.bound,
            proofs,
            self,
        )?;
        self.finish(checked).map_err(stage_error)
    }
    fn finish(&self, checked: Composed) -> Result<Reduced<'a>> {
        let expected = self
            .plan
            .schedule
            .stages()
            .first()
            .ok_or_else(internal)?
            .instances
            .iter()
            .filter_map(|i| match i.relation {
                schedule::Relation::Native { tag, .. } => Some(tag),
                _ => None,
            });
        check_records(expected.clone(), &checked.native)?;
        let mut sources = reserve(self.original.len())?;
        for c in &self.original {
            sources.push(stacked::Claim {
                sources: copy(&c.sources)?,
                point: copy(&c.point)?,
                layer_point: copy(&c.layer_point)?,
                value: c.value,
            });
        }
        for (tag, record) in expected.zip(&checked.native) {
            let ctx = NativeContext {
                stage: record.stage,
                layer: &record.layer,
                zero: &record.zero,
            };
            let final_value = self.component(tag, ctx, &record.point, &record.values)?;
            let (_, claims) = self.transform(tag, &record.point, final_value)?;
            sources
                .try_reserve(claims.len())
                .map_err(|_| GadgetError::Memory)?;
            sources.extend(claims);
        }
        Ok(Reduced {
            graph: checked.graph,
            sources,
            multiplicities: &self.bound.multiplicities,
            external: self.external,
        })
    }
}

fn check_records(
    expected: impl Iterator<Item = u32>,
    records: &[stages::NativeEvaluation],
) -> Result<()> {
    let mut expected = expected;
    for record in records {
        if expected.next() != Some(record.tag) || record.stage != 1 {
            return Err(internal());
        }
        if let Some(first) = records.first()
            && (record.layer != first.layer || record.zero != first.zero)
        {
            return Err(internal());
        }
    }
    if expected.next().is_some() {
        return Err(internal());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn f(v: u64) -> Fp2 {
        Fp2::from(Fp::reduce(v))
    }
    #[test]
    fn original_graph_projection_binds_family_and_every_coordinate() {
        let projection = Projection::new(
            0,
            [Source::Tensor(TensorId(3)), Source::Tensor(TensorId(9))].into_iter(),
            1,
            2,
        )
        .unwrap();
        let point = [f(2), f(3), f(5), f(7)];
        let claim = stacked::Claim {
            sources: vec![Source::Tensor(TensorId(3)), Source::Tensor(TensorId(9))],
            point: point[1..3].to_vec(),
            layer_point: point[3..].to_vec(),
            value: f(11),
        };
        assert_eq!(projection.value(&claim, &point, 1), Ok(f(11)));
        for fault in 0..5 {
            let mut bad = claim.clone();
            match fault {
                0 => bad.sources.swap(0, 1),
                1 => {
                    bad.sources.pop();
                }
                2 => bad.point[0] += Fp2::ONE,
                3 => bad.layer_point[0] += Fp2::ONE,
                4 => {
                    bad.point.pop();
                }
                _ => unreachable!(),
            }
            assert_eq!(
                projection.value(&bad, &point, 1),
                Err(GadgetError::Internal)
            );
        }
        let into_layer = Projection::new(0, claim.sources.iter().copied(), 2, 2).unwrap();
        let mut forged = claim.clone();
        forged.point = point[2..4].to_vec();
        assert_eq!(
            into_layer.value(&forged, &point, 1),
            Err(GadgetError::Internal)
        );
    }
    #[test]
    fn every_native_record_is_retained_once_in_its_own_fixed_stage_context() {
        let record = |tag| stages::NativeEvaluation {
            tag,
            stage: 1,
            layer: vec![f(3)],
            zero: vec![f(5)],
            point: vec![f(7)],
            values: vec![f(11)],
        };
        let records = vec![record(3), record(1), record(2)];
        assert_eq!(check_records([3, 1, 2].into_iter(), &records), Ok(()));
        for fault in 0..7 {
            let mut bad = records.clone();
            match fault {
                0 => {
                    bad.pop();
                }
                1 => bad.push(record(4)),
                2 => bad[1].tag = 3,
                3 => bad.swap(0, 1),
                4 => bad[0].stage = 2,
                5 => bad[1].layer[0] += Fp2::ONE,
                6 => bad[1].zero[0] += Fp2::ONE,
                _ => unreachable!(),
            }
            assert_eq!(
                check_records([3, 1, 2].into_iter(), &bad),
                Err(GadgetError::Internal)
            );
        }
    }
}

#[cfg(test)]
mod runtime_tests;

#[cfg(test)]
mod rejection_mapping_tests {
    use super::*;
    #[test]
    fn native_rejections_keep_their_gadget_instead_of_becoming_internal() {
        let gadget = crate::validation::GadgetId(23);
        assert_eq!(
            stage_error(GadgetError::Rejected { gadget }),
            StageError::Gadget { gadget }
        );
        assert_eq!(stage_error(GadgetError::Memory), StageError::Memory);
        assert_eq!(stage_error(GadgetError::Internal), StageError::Internal);
    }
}
