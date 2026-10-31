// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public requested-buffer geometry of the model leaf source (RFC-0018/0050).

use super::*;

/// Requested owned lookup-column/forest buffers, before canonical source construction.
/// This includes descriptor/domain capacities and own constructor scratch. A source's
/// simultaneous tile/gadget work, borrowed graph/model/engine, lent views, other proof
/// phases, runtime and allocator overhead are additional. No CPU route is admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelWitnessMemory {
    retained: u64,
    peak: u64,
}

fn add(a: u64, b: u64) -> Result<u64, LeafError> {
    a.checked_add(b).ok_or(LeafError::Memory)
}
fn array<T>(count: u64) -> Result<u64, LeafError> {
    let bytes = count
        .checked_mul(size_of::<T>() as u64)
        .ok_or(LeafError::Memory)?;
    if usize::try_from(count).is_err() || bytes > isize::MAX as u64 {
        return Err(LeafError::Memory);
    }
    Ok(bytes)
}
fn padded(mut domain: impl Iterator<Item = u32>) -> Result<u64, LeafError> {
    let variables = domain
        .try_fold(0u32, |sum, d| {
            sum.checked_add(d.checked_next_power_of_two()?.trailing_zeros())
        })
        .ok_or(LeafError::Memory)?;
    1u64.checked_shl(variables).ok_or(LeafError::Memory)
}
fn node_id(site: NodeRef, layer: usize, layers: &LayerMap) -> Result<NodeId, LeafError> {
    match site {
        NodeRef::Node(id) => Ok(id),
        NodeRef::Position(position) => {
            let layer = u32::try_from(layer).map_err(|_| LeafError::Memory)?;
            if layer >= layers.count || position >= layers.width {
                return own();
            }
            layers
                .width
                .checked_mul(layer)
                .and_then(|n| n.checked_add(layers.first))
                .and_then(|n| n.checked_add(position))
                .map(NodeId)
                .ok_or(LeafError::Memory)
        }
    }
}
impl ModelWitnessMemory {
    pub(super) fn new(
        layout: &LeafLayout<ModelColumn>,
        lowering: &Lowering,
        layers: &LayerMap,
        advice: &ModelAdviceLayout<'_>,
    ) -> Result<Self, LeafError> {
        let mut retained = array::<Vec<Vec<Column>>>(layout.sites().len() as u64)?;
        let mut rank = 0u64;
        let mut greedy = 0u64;
        let mut requests = 0u64;
        let mut forests = 0u64;
        for (s, site) in layout.sites().iter().enumerate() {
            if site.layers.is_empty() {
                return own();
            }
            retained = add(retained, array::<Vec<Column>>(site.layers.len() as u64)?)?;
            let mut forest = None;
            for (l, gadget) in site.layers.iter().enumerate() {
                let node = lowering
                    .graph
                    .nodes
                    .get(node_id(site.node, l, layers)?.0 as usize)
                    .ok_or(LeafError::Internal {
                        rule: "a source owner belongs to the lowering",
                    })?;
                let reads = layout.reads(s, l).ok_or(LeafError::Internal {
                    rule: "every model site has its ordered reads",
                })?;
                let softmax = matches!(node.op, tessera_ir::Op::SoftmaxRow { .. });
                retained = add(
                    retained,
                    array::<Column>(add(reads.len() as u64, u64::from(softmax))?)?,
                )?;
                requests = requests.max(gadget.lookups.len() as u64);
                for read in reads {
                    let request = gadget
                        .lookups
                        .iter()
                        .find(|r| {
                            core::iter::once(&r.lookup.key)
                                .chain(&r.lookup.value)
                                .any(|f| f.terms.iter().any(|(_, c)| *c == read.term))
                        })
                        .ok_or(LeafError::Internal {
                            rule: "a read belongs to a canonical lookup",
                        })?;
                    // Check every selected public axis before the infallible iterator below.
                    if read
                        .axes
                        .iter()
                        .any(|&axis| request.domain.get(axis).is_none())
                    {
                        return own();
                    }
                    let domain = || read.axes.iter().map(|&axis| request.domain[axis]);
                    let count = padded(domain())?;
                    rank = rank.max(read.axes.len() as u64);
                    retained = add(
                        retained,
                        add(array::<u32>(read.axes.len() as u64)?, array::<i64>(count)?)?,
                    )?;
                    if read.term.is_greedy_key() {
                        greedy = add(greedy, 1)?;
                    }
                    if l == 0 && read.term == ModelColumn::ExpKey && forest.is_none() {
                        if read.axes.len() != 3 {
                            return own();
                        }
                        let keys = request.domain[read.axes[2]];
                        if keys > 1 {
                            forest = Some(
                                count
                                    .checked_mul(
                                        (site.layers.len() as u64)
                                            .checked_next_power_of_two()
                                            .ok_or(LeafError::Memory)?,
                                    )
                                    .ok_or(LeafError::Memory)?,
                            );
                        }
                    }
                }
                if softmax {
                    let output = node
                        .outputs
                        .get(1)
                        .and_then(|id| lowering.graph.tensors.get(id.0 as usize))
                        .ok_or(LeafError::Internal {
                            rule: "a source owner has its canonical maximum output",
                        })?;
                    rank = rank.max(output.shape.len() as u64);
                    retained = add(
                        retained,
                        add(
                            array::<u32>(output.shape.len() as u64)?,
                            array::<i64>(padded(output.shape.iter().copied())?)?,
                        )?,
                    )?;
                }
            }
            if let Some(count) = forest {
                forests = add(forests, 1)?;
                retained = add(retained, array::<Goldilocks>(count)?)?;
            }
        }
        retained = add(retained, array::<Trees>(forests)?)?;
        if advice
            .tiles()
            .iter()
            .any(|tile| matches!(tile.tile, Tile::GreedyMaximum { .. }))
        {
            let tokens = lowering
                .graph
                .tensors
                .get(lowering.tokens.0 as usize)
                .and_then(|t| t.shape.first())
                .ok_or(LeafError::Internal {
                    rule: "the source lowering has its canonical token axis",
                })?;
            retained = add(
                retained,
                array::<i64>(u64::from(
                    tokens
                        .checked_next_power_of_two()
                        .ok_or(LeafError::Memory)?,
                ))?,
            )?;
        }
        // Conservative sum of own public temporaries; the complete caller also charges
        // the external TileWitness and its construction beside these owned buffers.
        let scratch = add(
            add(
                array::<core::ops::Range<u32>>(rank)?,
                array::<(usize, usize, usize)>(greedy)?,
            )?,
            array::<tessera_witness::TileLookup<'_>>(requests)?,
        )?;
        Ok(Self {
            retained,
            peak: add(retained, scratch)?,
        })
    }

    /// Requested retained column/forest buffers, including unused reserved descriptors.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }

    /// Upper bound for owned buffers and own constructor metadata, excluding source work.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_arrays_reject_address_and_product_overflow_without_allocation() {
        for (count, wide) in [
            ((isize::MAX as u64) + 1, false),
            (u64::MAX, true),
            (1u64 << 61, true),
        ] {
            let mut result = None;
            let measured = allocation_counter::measure(|| {
                result = Some(if wide {
                    array::<u64>(count)
                } else {
                    array::<u8>(count)
                })
            });
            assert_eq!(result.unwrap(), Err(LeafError::Memory));
            assert_eq!(measured.count_total, 0);
        }
        assert_eq!(padded([1, 3, 5].into_iter()).unwrap(), 32);
        assert_eq!(
            padded([1u32 << 31, 1u32 << 31, 4].into_iter()),
            Err(LeafError::Memory)
        );
        assert_eq!(padded([u32::MAX].into_iter()), Err(LeafError::Memory));
    }
}
