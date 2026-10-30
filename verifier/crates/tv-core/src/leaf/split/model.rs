// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical lookup forms from RFC-0009/0024/0045/0050. No proof data enters their derivation.
use super::{Affine, Column, Layer, Layout, Lookup, Read, Site, TokenAxis};
use crate::advice::{Auxiliary, Source};
use crate::claims::GadgetPolicy;
use crate::gadgets::{GadgetError, Plan, root::Root};
use crate::leaf::groups::TokenAxes;
use crate::lowering::{Graph, Node, NodeId, Op, TensorId};
use crate::selection::{self, Choice, Pattern, Selection};
use crate::validation::ClaimsError;
use tv_field::Fp;
use tv_tables::{TableId, TableKind};

fn internal() -> ClaimsError {
    ClaimsError::Internal
}
fn reserve<T>(n: usize) -> Result<Vec<T>, ClaimsError> {
    super::vector(n).map_err(|e| match e {
        super::Error::Memory => ClaimsError::Memory,
        _ => internal(),
    })
}
fn gadget(error: GadgetError) -> ClaimsError {
    match error {
        GadgetError::Memory => ClaimsError::Memory,
        _ => internal(),
    }
}
fn bits(n: u32) -> Result<u32, ClaimsError> {
    Ok(u32::BITS.saturating_sub(n.checked_sub(1).ok_or_else(internal)?.leading_zeros()))
}
fn variables(graph: &Graph, id: TensorId) -> Result<u32, ClaimsError> {
    let dims = graph.tensor(id).ok_or_else(internal)?.shape.dims();
    if let Some(key_bits) = graph.attention_key_bits(id) {
        let prefix = dims
            .get(..dims.len().checked_sub(1).ok_or_else(internal)?)
            .ok_or_else(internal)?;
        return prefix
            .iter()
            .try_fold(0u32, |sum, &d| {
                sum.checked_add(bits(d)?).ok_or_else(internal)
            })?
            .checked_add(key_bits)
            .ok_or_else(internal);
    }
    dims.iter().try_fold(0u32, |sum, &d| {
        sum.checked_add(bits(d)?).ok_or_else(internal)
    })
}
/// A domain's token axis among its block variables (RFC-0054 'The token axis'): the variables of
/// the axes after it come first (RFC-0045 'Blocks').
pub(crate) fn token(
    graph: &Graph,
    axes: &TokenAxes,
    id: TensorId,
) -> Result<Option<TokenAxis>, ClaimsError> {
    let Some(axis) = axes.get(id).ok_or_else(internal)? else {
        return Ok(None);
    };
    let dims = graph.tensor(id).ok_or_else(internal)?.shape.dims();
    let axis = usize::from(axis);
    let key_bits = graph.attention_key_bits(id);
    let last_axis = dims.len().checked_sub(1).ok_or_else(internal)?;
    let low = dims
        .get(axis.checked_add(1).ok_or_else(internal)?..)
        .ok_or_else(internal)?
        .iter()
        .enumerate()
        .try_fold(0u32, |sum, (offset, &d)| {
            let at = axis
                .checked_add(1)
                .and_then(|x| x.checked_add(offset))
                .ok_or_else(internal)?;
            let v = match key_bits {
                Some(kb) if at == last_axis => kb,
                _ => bits(d)?,
            };
            sum.checked_add(v).ok_or_else(internal)
        })?;
    Ok(Some(TokenAxis {
        low,
        bits: bits(*dims.get(axis).ok_or_else(internal)?)?,
    }))
}
fn power(n: u32) -> Result<Fp, ClaimsError> {
    Ok(Fp::reduce(1u64.checked_shl(n).ok_or_else(internal)?))
}
fn affine(constant: Fp, terms: &[(Fp, Column, u32)]) -> Result<Affine, ClaimsError> {
    let mut out = reserve(terms.len())?;
    for &(coefficient, column, vars) in terms {
        let mut coordinates = reserve(usize::try_from(vars).map_err(|_| internal())?)?;
        coordinates.extend(0..vars);
        out.push((
            coefficient,
            Read {
                column,
                coordinates,
            },
        ));
    }
    Ok(Affine {
        constant,
        terms: out,
    })
}
fn one(column: Column, vars: u32) -> Result<Affine, ClaimsError> {
    affine(Fp::ZERO, &[(Fp::ONE, column, vars)])
}
fn request(
    table: TableId,
    (variables, token): (u32, Option<TokenAxis>),
    key: Affine,
    value: Option<Affine>,
) -> Lookup {
    Lookup {
        table,
        variables,
        key,
        value,
        token,
    }
}
fn requant(
    graph: &Graph,
    axes: &TokenAxes,
    selected: &Selection,
    choice: &Choice,
    node: &Node,
) -> Result<Layer, ClaimsError> {
    let input = *node.inputs().first().ok_or_else(internal)?;
    let vars = variables(graph, input)?;
    let domain = (vars, token(graph, axes, input)?);
    let shape = graph.tensor(input).ok_or_else(internal)?.shape;
    let channels = bits(*shape.dims().last().ok_or_else(internal)?)?;
    let table = selected
        .tables
        .get(
            usize::try_from(choice.tables.last().ok_or_else(internal)?.0)
                .map_err(|_| internal())?,
        )
        .ok_or_else(internal)?;
    let (final_bits, classes) = match &table.kind {
        TableKind::Requant { b, r_by_class, .. } | TableKind::Silu { b, r_by_class, .. } => {
            (*b, r_by_class.len())
        }
        _ => return Err(internal()),
    };
    if classes > 1 {
        let class = *node.inputs().get(1).ok_or_else(internal)?;
        if graph.tensor(class).ok_or_else(internal)?.shape.dims()
            != shape
                .dims()
                .last()
                .map(core::slice::from_ref)
                .ok_or_else(internal)?
        {
            return Err(internal());
        }
    }
    let width = selected.key_bits(choice)?;
    let off = power(width.checked_sub(1).ok_or_else(internal)?)?;
    let mut lookups = reserve(choice.tables.len())?;
    let mut ids = choice.tables.iter();
    let mut push = |key: Affine, value: Option<Affine>| -> Result<(), ClaimsError> {
        lookups.push(request(
            *ids.next().ok_or_else(internal)?,
            domain,
            key,
            value,
        ));
        Ok(())
    };
    let class = |terms: &mut Vec<(Fp, Column, u32)>| -> Result<(), ClaimsError> {
        if classes > 1 {
            terms.push((power(u32::from(final_bits))?, Column::Input(1), channels));
        }
        Ok(())
    };
    match choice.pattern {
        Pattern::Requant1 | Pattern::Lookup => {
            let mut terms = reserve(2)?;
            terms.push((Fp::ONE, Column::Input(0), vars));
            class(&mut terms)?;
            push(affine(off, &terms)?, Some(one(Column::Output(0), vars)?))?;
        }
        Pattern::RequantSplit { split } => {
            let low = Column::Auxiliary(Auxiliary::RequantLow);
            push(one(low, vars)?, None)?;
            let inv = power(u32::from(split))?.inverse().ok_or_else(internal)?;
            let mut terms = reserve(3)?;
            terms.extend([(inv, Column::Input(0), vars), (-inv, low, vars)]);
            class(&mut terms)?;
            push(
                affine(off * inv, &terms)?,
                Some(one(Column::Output(0), vars)?),
            )?;
        }
        Pattern::RequantWide { split, middle } => {
            let low = Column::Auxiliary(Auxiliary::RequantLow);
            let mid = Column::Auxiliary(Auxiliary::RequantMiddle);
            let saturation = Column::Auxiliary(Auxiliary::RequantSaturation);
            if split > 0 {
                push(one(low, vars)?, None)?;
            }
            push(one(mid, vars)?, None)?;
            let inv = power(
                u32::from(split)
                    .checked_add(u32::from(middle))
                    .ok_or_else(internal)?,
            )?
            .inverse()
            .ok_or_else(internal)?;
            let mut top = reserve(3)?;
            top.push((inv, Column::Input(0), vars));
            if split > 0 {
                top.push((-inv, low, vars));
            }
            top.push((-power(u32::from(split))? * inv, mid, vars));
            push(affine(off * inv, &top)?, None)?;
            let mut sign = reserve(top.len())?;
            sign.extend(top.iter().map(|&(a, c, v)| (a * Fp::reduce(2), c, v)));
            push(
                affine(Fp::reduce(2) * off * inv, &sign)?,
                Some(one(saturation, vars)?),
            )?;
            let mut terms = reserve(3)?;
            terms.extend([
                (Fp::ONE, mid, vars),
                (power(u32::from(middle))?, saturation, vars),
            ]);
            class(&mut terms)?;
            push(
                affine(
                    power(u32::from(middle).checked_add(1).ok_or_else(internal)?)?,
                    &terms,
                )?,
                Some(one(Column::Output(0), vars)?),
            )?;
        }
        _ => return Err(internal()),
    }
    if ids.next().is_some() {
        return Err(internal());
    }
    Ok(Layer {
        node: choice.node,
        lookups,
    })
}
fn root(graph: &Graph, axes: &TokenAxes, site: &Root<'_>) -> Result<Layer, ClaimsError> {
    let node = graph.node(site.node()).ok_or_else(internal)?;
    let output = *node.outputs().first().ok_or_else(internal)?;
    let vars = variables(graph, output)?;
    let domain = (vars, token(graph, axes, output)?);
    let width = u32::from(*site.range_bits().first().ok_or_else(internal)?);
    let low = width
        .checked_sub(1)
        .ok_or_else(internal)?
        .checked_div(8)
        .ok_or_else(internal)?;
    let mut keys = reserve(site.tables().len())?;
    keys.push(one(Column::Output(0), vars)?);
    for index in 0..3 {
        keys.push(one(Column::Auxiliary(Auxiliary::RootInput(index)), vars)?);
    }
    let inv = power(48)?.inverse().ok_or_else(internal)?;
    let mut terms = reserve(4)?;
    terms.push((inv, Column::RootInput, vars));
    for index in 0..3u32 {
        terms.push((
            -power(index.checked_mul(16).ok_or_else(internal)?)? * inv,
            Column::Auxiliary(Auxiliary::RootInput(index)),
            vars,
        ));
    }
    keys.push(affine(Fp::ZERO, &terms)?);
    for index in 0..low {
        keys.push(one(Column::Auxiliary(Auxiliary::RootLow(index)), vars)?);
    }
    let inv = power(low.checked_mul(8).ok_or_else(internal)?)?
        .inverse()
        .ok_or_else(internal)?;
    let mut terms = reserve(
        usize::try_from(low)
            .map_err(|_| internal())?
            .checked_add(1)
            .ok_or_else(internal)?,
    )?;
    terms.push((inv, Column::Output(0), vars));
    for index in 0..low {
        terms.push((
            -power(index.checked_mul(8).ok_or_else(internal)?)? * inv,
            Column::Auxiliary(Auxiliary::RootLow(index)),
            vars,
        ));
    }
    keys.push(affine(Fp::ZERO, &terms)?);
    let start = 5usize
        .checked_add(usize::try_from(low).map_err(|_| internal())?)
        .ok_or_else(internal)?;
    for source in site.sources().get(start..).ok_or_else(internal)? {
        let Source::Auxiliary(owner, role) = *source else {
            return Err(internal());
        };
        if owner != site.node() {
            return Err(internal());
        }
        keys.push(one(Column::Auxiliary(role), vars)?);
    }
    if keys.len() != site.tables().len() {
        return Err(internal());
    }
    let mut lookups = reserve(keys.len())?;
    for (&table, key) in site.tables().iter().zip(keys) {
        lookups.push(request(table, domain, key, None));
    }
    Ok(Layer {
        node: site.node(),
        lookups,
    })
}
fn softmax(
    graph: &Graph,
    axes: &TokenAxes,
    policy: GadgetPolicy,
    choice: &Choice,
    node: &Node,
    sigma: u8,
) -> Result<Layer, ClaimsError> {
    let scores = *node.inputs().first().ok_or_else(internal)?;
    let rows = *node.outputs().get(1).ok_or_else(internal)?;
    let score = variables(graph, scores)?;
    let row = variables(graph, rows)?;
    // The scores' token axis is their query axis, and so is the reciprocals' (RFC-0054 'Rows').
    let scored = (score, token(graph, axes, scores)?);
    let rowed = (row, token(graph, axes, rows)?);
    let width = u32::from(sigma).checked_add(18).ok_or_else(internal)?;
    let limb = u32::from(policy.limb_bits);
    let low = width
        .checked_add(limb.checked_sub(1).ok_or_else(internal)?)
        .and_then(|v| v.checked_div(limb))
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(internal)?;
    let mut lookups = reserve(choice.tables.len())?;
    let mut ids = choice.tables.iter();
    lookups.push(request(
        *ids.next().ok_or_else(internal)?,
        scored,
        one(Column::SoftmaxKey, score)?,
        Some(one(Column::Output(0), score)?),
    ));
    lookups.push(request(
        *ids.next().ok_or_else(internal)?,
        rowed,
        one(Column::Output(1), row)?,
        None,
    ));
    for a in [true, false] {
        let column = |index| {
            Column::Auxiliary(if a {
                Auxiliary::SoftmaxA(index)
            } else {
                Auxiliary::SoftmaxB(index)
            })
        };
        for index in 0..low {
            lookups.push(request(
                *ids.next().ok_or_else(internal)?,
                rowed,
                one(column(index), row)?,
                None,
            ));
        }
        let inv = power(low.checked_mul(limb).ok_or_else(internal)?)?
            .inverse()
            .ok_or_else(internal)?;
        let mut terms = reserve(
            usize::try_from(low)
                .map_err(|_| internal())?
                .checked_add(1)
                .ok_or_else(internal)?,
        )?;
        terms.push((
            inv,
            if a {
                Column::SoftmaxA
            } else {
                Column::SoftmaxB
            },
            row,
        ));
        for index in 0..low {
            terms.push((
                -power(index.checked_mul(limb).ok_or_else(internal)?)? * inv,
                column(index),
                row,
            ));
        }
        lookups.push(request(
            *ids.next().ok_or_else(internal)?,
            rowed,
            affine(Fp::ZERO, &terms)?,
            None,
        ));
    }
    if ids.next().is_some() {
        return Err(internal());
    }
    Ok(Layer {
        node: choice.node,
        lookups,
    })
}
fn producer(graph: &Graph, tensor: TensorId) -> Result<usize, ClaimsError> {
    graph
        .nodes()
        .iter()
        .position(|n| n.outputs().contains(&tensor))
        .ok_or_else(internal)
}

/// Canonical lookup-site descriptors of the independently lowered current model.
/// Sites use local graph/auxiliary identities and the virtual tensors named by RFC-0009.
/// Each request carries its lookup tensor's token axis (RFC-0054 'The token axis').
/// This is public metadata, never a proof-supplied form or source registry.
///
/// # Errors
/// Public selection errors; `Memory` on reserve failure; `Internal` for inconsistent own
/// template/form geometry, or a graph the token-axis rules refuse ([`TokenAxes::derive`]).
/// `Layout::new` additionally checks cross-layer read consistency.
pub fn model_sites(graph: &Graph, policy: GadgetPolicy) -> Result<Vec<Site>, ClaimsError> {
    let axes = TokenAxes::derive(graph).map_err(|e| match e {
        crate::leaf::groups::AxisError::Memory => ClaimsError::Memory,
        _ => internal(),
    })?;
    sites(graph, policy, &axes)
}

/// [`model_sites`] with the graph's token axes already derived.
pub(crate) fn sites(
    graph: &Graph,
    policy: GadgetPolicy,
    axes: &TokenAxes,
) -> Result<Vec<Site>, ClaimsError> {
    let layer_count = crate::schedule::template_layers(graph).map_err(|e| match e {
        crate::schedule::ScheduleError::Memory => ClaimsError::Memory,
        _ => internal(),
    })?;
    let selected = selection::derive(graph, policy)?;
    let plan = Plan::derive(graph, policy)?;
    let mut layers = reserve(selected.choices.len())?;
    for choice in &selected.choices {
        let node = graph.node(choice.node).ok_or_else(internal)?;
        let layer = match *node.op() {
            Op::Requant { .. } | Op::TableMap { .. } => {
                requant(graph, axes, &selected, choice, node)?
            }
            Op::SoftmaxRow { sigma, .. } => softmax(graph, axes, policy, choice, node, sigma)?,
            Op::RsqrtRow { .. } | Op::OneHot { .. } => continue,
            _ => return Err(internal()),
        };
        layers.push(layer);
    }
    for site in plan.roots() {
        layers.push(root(graph, axes, &site.map_err(gadget)?)?);
    }
    let first = producer(graph, *graph.checkpoints().first().ok_or_else(internal)?)?
        .checked_add(1)
        .ok_or_else(internal)?;
    let end = producer(graph, *graph.checkpoints().last().ok_or_else(internal)?)?
        .checked_add(1)
        .ok_or_else(internal)?;
    let span = end.checked_sub(first).ok_or_else(internal)?;
    let count = usize::try_from(layer_count).map_err(|_| internal())?;
    if span == 0 || span.checked_rem(count) != Some(0) {
        return Err(internal());
    }
    let width = span.checked_div(count).ok_or_else(internal)?;
    let representative = |node: NodeId| -> Option<usize> {
        let index = usize::try_from(node.0).ok()?;
        if index >= first && index < end {
            first.checked_add(index.checked_sub(first)?.checked_rem(width)?)
        } else {
            Some(index)
        }
    };
    layers.sort_unstable_by_key(|layer| (representative(layer.node), layer.node.0));
    let mut sites: Vec<Site> = reserve(layers.len())?;
    let mut previous = None;
    for layer in layers {
        let owner = representative(layer.node).ok_or_else(internal)?;
        if previous == Some(owner) {
            let site = sites.last_mut().ok_or_else(internal)?;
            site.layers
                .try_reserve(1)
                .map_err(|_| ClaimsError::Memory)?;
            site.layers.push(layer);
        } else {
            let mut real = reserve(count)?;
            real.push(layer);
            sites.push(Site { layers: real });
            previous = Some(owner);
        }
    }
    Ok(sites)
}
impl Layout {
    /// Derive and compile all canonical lookup forms of a public model graph (SP-7).
    /// Virtual inputs remain explicit obligations in the result of `bind`.
    ///
    /// # Errors
    /// Public selection errors, `Memory`, or `Internal` for inconsistent own model geometry.
    pub fn derive(graph: &Graph, policy: GadgetPolicy) -> Result<Self, ClaimsError> {
        Self::new(&model_sites(graph, policy)?).map_err(|e| match e {
            super::Error::Memory => ClaimsError::Memory,
            _ => internal(),
        })
    }
}
