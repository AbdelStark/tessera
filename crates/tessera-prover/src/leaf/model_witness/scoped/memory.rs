// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical two-pass scoped source owners, excluding graph/tile callback work.

use super::*;
use crate::leaf::model_witness::memory::{add, array, node_id};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScopedGroupMemory {
    pub(crate) construction: u64,
    pub(crate) retained: u64,
    /// All owned merge block copies can coexist with the current input.
    pub(crate) blocks: u64,
}

#[derive(Clone, Copy)]
struct BuildMemory {
    retained: u64,
    peak: u64,
}

fn selected_memory(
    layout: &LeafLayout<ModelColumn>,
    lowering: &Lowering,
    layers: &LayerMap,
    selection: Selection<'_>,
) -> Result<BuildMemory, LeafError> {
    let mut retained = array::<Vec<Vec<Column>>>(layout.sites().len() as u64)?;
    let mut rank = 0u64;
    let mut greedy = 0u64;
    let mut requests = 0u64;
    for (s, site) in layout.sites().iter().enumerate() {
        if site.layers.is_empty() {
            return own();
        }
        retained = add(retained, array::<Vec<Column>>(site.layers.len() as u64)?)?;
        for (l, gadget) in site.layers.iter().enumerate() {
            let node = lowering
                .graph
                .nodes
                .get(node_id(site.node, l, layers)?.0 as usize)
                .ok_or(LeafError::Internal {
                    rule: "a scoped source owner belongs to the lowering",
                })?;
            let reads = layout.reads(s, l).ok_or(LeafError::Internal {
                rule: "a scoped source has every canonical ordered read",
            })?;
            let softmax = matches!(node.op, tessera_ir::Op::SoftmaxRow { .. });
            retained = add(
                retained,
                array::<Column>(add(reads.len() as u64, u64::from(softmax))?)?,
            )?;
            requests = requests.max(gadget.lookups.len() as u64);
            for read in reads {
                greedy = add(greedy, u64::from(read.term.is_greedy_key()))?;
                rank = rank.max(read.axes.len() as u64);
                let request = gadget
                    .lookups
                    .iter()
                    .find(|r| {
                        core::iter::once(&r.lookup.key)
                            .chain(&r.lookup.value)
                            .any(|f| f.terms.iter().any(|(_, c)| *c == read.term))
                    })
                    .ok_or(LeafError::Internal {
                        rule: "a scoped read has its canonical lookup",
                    })?;
                let vars = read.axes.iter().try_fold(0u32, |v, &axis| {
                    let d = request.domain.get(axis).ok_or(LeafError::Internal {
                        rule: "a scoped source axis belongs to its lookup domain",
                    })?;
                    v.checked_add(
                        d.checked_next_power_of_two()
                            .ok_or(LeafError::Memory)?
                            .trailing_zeros(),
                    )
                    .ok_or(LeafError::Memory)
                })?;
                if let Some(p) = selection.projection(layout, s, l, read.term, vars)? {
                    retained = add(
                        retained,
                        add(
                            array::<u32>(read.axes.len() as u64)?,
                            array::<i64>(p.len() as u64)?,
                        )?,
                    )?;
                }
            }
            if softmax
                && matches!(
                    selection,
                    Selection::Read {
                        term: ModelColumn::Auxiliary(tessera_compiler::AuxColumn::Maximum),
                        ..
                    }
                )
            {
                let shape = &lowering
                    .graph
                    .tensors
                    .get(
                        node.outputs
                            .get(1)
                            .ok_or(LeafError::Internal {
                                rule: "a scoped Softmax source has its maximum output",
                            })?
                            .0 as usize,
                    )
                    .ok_or(LeafError::Internal {
                        rule: "a scoped maximum output belongs to the graph",
                    })?
                    .shape;
                rank = rank.max(shape.len() as u64);
                if let Some(p) = selection.projection(
                    layout,
                    s,
                    l,
                    ModelColumn::Auxiliary(tessera_compiler::AuxColumn::Maximum),
                    variables(shape).ok_or(LeafError::Memory)?,
                )? {
                    retained = add(
                        retained,
                        add(
                            array::<u32>(shape.len() as u64)?,
                            array::<i64>(p.len() as u64)?,
                        )?,
                    )?;
                }
            }
        }
    }
    let mut forests = 0u64;
    let mut positions = 0u64;
    for (s, site) in layout.sites().iter().enumerate() {
        let Some(domain) = forest_domain(layout, s)? else {
            continue;
        };
        let vars = variables(&domain)
            .ok_or(LeafError::Memory)?
            .checked_add(
                site.layers
                    .len()
                    .checked_next_power_of_two()
                    .ok_or(LeafError::Memory)?
                    .trailing_zeros(),
            )
            .ok_or(LeafError::Memory)?;
        let held = match selection {
            Selection::Group(grouped, _) => {
                positions = u64::from(grouped.groups().bits());
                Some(
                    vars.checked_sub(u32::from(grouped.groups().bits()))
                        .ok_or(LeafError::Memory)?,
                )
            }
            Selection::Forest { site } if site == s => Some(vars),
            Selection::Read { .. } | Selection::Forest { .. } => None,
            Selection::All => return own(),
        };
        if let Some(vars) = held {
            forests = add(forests, 1)?;
            retained = add(
                retained,
                array::<Goldilocks>(1u64.checked_shl(vars).ok_or(LeafError::Memory)?)?,
            )?;
        }
    }
    retained = add(retained, array::<Trees>(forests)?)?;
    // A skipped read still creates its temporary domain. Slot/scatter ranges,
    // greedy's complete canonical map and the current tile's request descriptors
    // also coexist with selected buffers. forest_domain currently collects 3 Options.
    let scratch = add(
        add(array::<u32>(rank)?, array::<core::ops::Range<u32>>(rank)?)?,
        add(
            array::<(usize, usize, usize)>(greedy)?,
            add(
                array::<tessera_witness::TileLookup<'_>>(requests)?,
                add(array::<usize>(positions)?, array::<Option<u32>>(3)?)?,
            )?,
        )?,
    )?;
    Ok(BuildMemory {
        retained,
        peak: add(retained, scratch)?,
    })
}

impl ScopedGroupMemory {
    /// Source and group layout bounds only; graph/cache/tile work is additional.
    pub(crate) fn new(
        layout: &LeafLayout<ModelColumn>,
        lowering: &Lowering,
        layers: &LayerMap,
        grouped: &GroupedLayout,
        h: usize,
    ) -> Result<Self, LeafError> {
        let layout_memory = grouped.group_memory(h)?;
        let group = grouped.group(h)?;
        let build = selected_memory(layout, lowering, layers, Selection::Group(grouped, &group))?;
        let mut field = 0;
        let mut splits = 0;
        for read in group.reads() {
            let shape = grouped
                .polynomials()
                .get(read.polynomial)
                .ok_or(LeafError::Internal {
                    rule: "a scoped memory read has its original polynomial",
                })?;
            let free = shape
                .vars
                .checked_sub(shape.group_vars.len())
                .ok_or(LeafError::Memory)?;
            splits = add(
                splits,
                array::<Goldilocks>(
                    1u64.checked_shl(u32::try_from(free).map_err(|_| LeafError::Memory)?)
                        .ok_or(LeafError::Memory)?,
                )?,
            )?;
        }
        for instance in group.instances() {
            let rows = 1u64.checked_shl(instance.height).ok_or(LeafError::Memory)?;
            let value = instance
                .blocks
                .first()
                .and_then(|b| {
                    layout
                        .sites()
                        .get(b.site)
                        .and_then(|s| s.layers.get(b.layer))
                        .and_then(|g| g.lookups.get(b.lookup))
                })
                .ok_or(LeafError::Internal {
                    rule: "a scoped instance has a canonical lookup",
                })?
                .lookup
                .value
                .is_some();
            field = add(
                field,
                array::<Goldilocks>(
                    rows.checked_mul(1 + u64::from(value))
                        .ok_or(LeafError::Memory)?,
                )?,
            )?;
        }
        let mut forests = 0;
        let mut forest_bytes = 0;
        for (s, site) in layout.sites().iter().enumerate() {
            let Some(domain) = forest_domain(layout, s)? else {
                continue;
            };
            forests = add(forests, 1)?;
            let vars = variables(&domain)
                .ok_or(LeafError::Memory)?
                .checked_add(
                    site.layers
                        .len()
                        .checked_next_power_of_two()
                        .ok_or(LeafError::Memory)?
                        .trailing_zeros(),
                )
                .and_then(|v| v.checked_sub(u32::from(grouped.groups().bits())))
                .ok_or(LeafError::Memory)?;
            forest_bytes = add(
                forest_bytes,
                array::<Goldilocks>(1u64.checked_shl(vars).ok_or(LeafError::Memory)?)?,
            )?;
        }
        let transferred = add(
            add(field, splits)?,
            add(
                array::<(
                    tessera_tables::TableId,
                    SecretBuf<Goldilocks>,
                    Option<SecretBuf<Goldilocks>>,
                )>(group.instances().len() as u64)?,
                add(
                    array::<SecretBuf<Goldilocks>>(group.reads().len() as u64)?,
                    array::<SecretBuf<Goldilocks>>(forests)?,
                )?,
            )?,
        )?;
        let blocks = add(
            add(splits, forest_bytes)?,
            array::<(usize, u64, tessera_codec::Secret<Vec<Goldilocks>>)>(add(
                group.reads().len() as u64,
                forests,
            )?)?,
        )?;
        // The selected raw i64 buffers coexist with converted lookup/split buffers;
        // forest backing moves instead of copying, and is counted only once here.
        Ok(ScopedGroupMemory {
            construction: add(layout_memory.peak_bytes(), add(build.peak, transferred)?)?,
            retained: add(
                layout_memory.output_bytes(),
                add(transferred, forest_bytes)?,
            )?,
            blocks,
        })
    }
}

/// Maximum one-original source/copy overlap, excluding retained merge/kernel owners.
pub(crate) fn original_peak(
    layout: &LeafLayout<ModelColumn>,
    lowering: &Lowering,
    layers: &LayerMap,
    grouped: &GroupedLayout,
) -> Result<u64, LeafError> {
    let mut peak = 0;
    let mut polynomial = 0;
    for (s, site) in layout.sites().iter().enumerate() {
        for l in 0..site.layers.len() {
            for read in layout.reads(s, l).ok_or(LeafError::Internal {
                rule: "an original source has its canonical ordered reads",
            })? {
                let shape = grouped
                    .polynomials()
                    .get(polynomial)
                    .ok_or(LeafError::Internal {
                        rule: "an original source has its canonical polynomial identity",
                    })?;
                let count = 1u64
                    .checked_shl(u32::try_from(shape.vars).map_err(|_| LeafError::Memory)?)
                    .ok_or(LeafError::Memory)?;
                let source = selected_memory(
                    layout,
                    lowering,
                    layers,
                    Selection::Read {
                        site: s,
                        layer: l,
                        term: read.term,
                    },
                )?;
                // Conservative coexistence: all raw dependencies and both the
                // returned field polynomial and its clearing seal_copy owner.
                peak = peak.max(add(
                    source.peak,
                    array::<Goldilocks>(count.checked_mul(2).ok_or(LeafError::Memory)?)?,
                )?);
                polynomial += 1;
            }
        }
        if let Some(domain) = forest_domain(layout, s)? {
            let count = (1u64
                .checked_shl(variables(&domain).ok_or(LeafError::Memory)?)
                .ok_or(LeafError::Memory)?)
            .checked_mul(
                site.layers
                    .len()
                    .checked_next_power_of_two()
                    .ok_or(LeafError::Memory)? as u64,
            )
            .ok_or(LeafError::Memory)?;
            let source = selected_memory(layout, lowering, layers, Selection::Forest { site: s })?;
            // build already includes its original field forest; seal_copy adds one.
            peak = peak.max(add(source.peak, array::<Goldilocks>(count)?)?);
        }
    }
    if polynomial != grouped.polynomials().len() {
        return own();
    }
    Ok(peak)
}

/// One original/native column lease's construction, in addition to native state.
pub(crate) fn native_peak(
    layout: &LeafLayout<ModelColumn>,
    lowering: &Lowering,
    layers: &LayerMap,
    advice: &ModelAdviceLayout<'_>,
) -> Result<u64, LeafError> {
    let mut peak = 0;
    for (s, site) in layout.sites().iter().enumerate() {
        for l in 0..site.layers.len() {
            let node = lowering
                .graph
                .nodes
                .get(node_id(site.node, l, layers)?.0 as usize)
                .ok_or(LeafError::Internal {
                    rule: "a scoped source owner belongs to the lowering",
                })?;
            for term in layout
                .reads(s, l)
                .ok_or(LeafError::Internal {
                    rule: "a native source has its canonical ordered reads",
                })?
                .iter()
                .map(|r| r.term)
                .chain(
                    matches!(node.op, tessera_ir::Op::SoftmaxRow { .. })
                        .then_some(ModelColumn::Auxiliary(tessera_compiler::AuxColumn::Maximum)),
                )
            {
                let source = selected_memory(
                    layout,
                    lowering,
                    layers,
                    Selection::Read {
                        site: s,
                        layer: l,
                        term,
                    },
                )?;
                peak = peak.max(source.peak.max(source.retained));
            }
        }
    }
    if advice
        .tiles()
        .iter()
        .any(|t| matches!(t.tile, Tile::GreedyMaximum { .. }))
    {
        let count = lowering
            .graph
            .tensors
            .get(lowering.tokens.0 as usize)
            .and_then(|t| t.shape.first())
            .and_then(|n| n.checked_next_power_of_two())
            .ok_or(LeafError::Memory)?;
        peak = peak.max(array::<i64>(u64::from(count))?);
    }
    Ok(peak)
}

/// Canonical source-only maxima; complete physical composition remains mandatory.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScopedSourceMemory {
    pub(crate) group_construction: u64,
    pub(crate) group_retained: u64,
    pub(crate) group_blocks: u64,
    pub(crate) original: u64,
    pub(crate) native: u64,
    pub(crate) planning: u64,
}
impl ScopedSourceMemory {
    pub(crate) fn new(
        layout: &LeafLayout<ModelColumn>,
        lowering: &Lowering,
        layers: &LayerMap,
        grouped: &GroupedLayout,
        advice: &ModelAdviceLayout<'_>,
    ) -> Result<Self, LeafError> {
        let mut result = Self {
            group_construction: 0,
            group_retained: 0,
            group_blocks: 0,
            original: original_peak(layout, lowering, layers, grouped)?,
            native: native_peak(layout, lowering, layers, advice)?,
            planning: 0,
        };
        for h in 0..grouped.count() {
            let group = ScopedGroupMemory::new(layout, lowering, layers, grouped, h)?;
            result.group_construction = result.group_construction.max(group.construction);
            result.group_retained = result.group_retained.max(group.retained);
            result.group_blocks = result.group_blocks.max(group.blocks);
            result.planning = result.planning.max(add(
                grouped.group_memory(h)?.peak_bytes(),
                array::<Option<u32>>(3)?,
            )?);
        }
        Ok(result)
    }
    /// Conservative source overlap across group, P2b and native construction.
    /// Proving kernels, outputs, callback/cache owners and all other phases are extra.
    pub(crate) fn peak(self) -> Result<u64, LeafError> {
        Ok(self
            .group_construction
            .max(add(self.group_retained, self.group_blocks)?)
            .max(self.original)
            .max(self.native))
    }
}
