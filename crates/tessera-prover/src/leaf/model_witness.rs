// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Resident CPU columns for the complete model leaf, borrowing canonical witness tiles.
use super::{LeafError, LeafLayout, ModelColumn, SiteTensors, StageForest, internal};
use core::fmt;
use tessera_codec::SecretBuf;
use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_compiler::{Lowering, ModelAdviceLayout, Share, Tile};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_gadgets::Affine;
use tessera_ir::{Node, NodeId};
use tessera_sumcheck::Forest;
use tessera_witness::WitnessSource;

struct Column {
    name: ModelColumn,
    domain: Vec<u32>,
    values: SecretBuf<i64>,
    written: usize,
}
struct Trees {
    site: NodeRef,
    height: usize,
    /// The leaf table's lowest variable of the layer index: the heads, queries and keys of one
    /// layer take the variables below it.
    layers: u32,
    leaves: SecretBuf<Goldilocks>,
}

/// The current model's padded lookup sources and Softmax forests, in clearing buffers.
/// Virtual columns have no advice positions and remain native reduction obligations.
/// This resident CPU adapter materializes the leaf columns; tree streaming is #134.
/// Its debug form shows only counts. Values are lent under the same borrower rules as
/// `TileWitness` (05 §3, OBS-1/6).
pub struct ModelWitness {
    // shortcut: materialize CPU leaf columns; add bounded tree streaming under #134.
    columns: Vec<Vec<Vec<Column>>>,
    forests: Vec<Trees>,
    maximum: Option<(tessera_ir::GadgetId, SecretBuf<i64>)>,
}
impl fmt::Debug for ModelWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelWitness")
            .field("sites", &self.columns.len())
            .field("forests", &self.forests.len())
            .finish_non_exhaustive()
    }
}
fn own<T>() -> Result<T, LeafError> {
    internal("model leaf columns cover every canonical source at its public coordinates")
}
fn variables(shape: &[u32]) -> Option<u32> {
    shape.iter().try_fold(0u32, |v, &d| {
        v.checked_add(d.checked_next_power_of_two()?.trailing_zeros())
    })
}
fn elements(shape: &[u32]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
}
fn buffer<T: Copy>(len: usize, value: T) -> Result<SecretBuf<T>, LeafError> {
    let mut out = SecretBuf::try_with_capacity(len).map_err(|_| LeafError::Memory)?;
    out.resize(len, value);
    Ok(out)
}
fn tensor(column: ModelColumn, node: &Node) -> Option<tessera_ir::TensorId> {
    match column {
        ModelColumn::Input(i) => node.inputs.get(usize::from(i)).copied(),
        ModelColumn::Output(i) => node.outputs.get(usize::from(i)).copied(),
        _ => None,
    }
}
fn owner(layers: &LayerMap, site: NodeRef, layer: usize) -> NodeId {
    match site {
        NodeRef::Node(n) => n,
        NodeRef::Position(p) => layers.node(p, layer as u32),
    }
}
fn virtual_column(c: ModelColumn) -> bool {
    matches!(
        c,
        ModelColumn::RootInput | ModelColumn::ExpKey | ModelColumn::Slack(_)
    )
}

/// Copy one canonical tile share into a per-axis padded column. All indices and branches
/// here depend only on public shapes/ranges, never on the copied values.
fn scatter<T: Copy>(
    out: &mut [i64],
    domain: &[u32],
    ranges: &[core::ops::Range<u32>],
    values: &[T],
    lift: fn(T) -> i64,
) -> Result<(), LeafError> {
    if ranges.len() != domain.len()
        || ranges
            .iter()
            .zip(domain)
            .any(|(r, &d)| r.start > r.end || r.end > d)
    {
        return own();
    }
    let count = ranges
        .iter()
        .try_fold(1usize, |n, r| n.checked_mul((r.end - r.start) as usize))
        .ok_or(LeafError::Memory)?;
    if count != values.len() {
        return own();
    }
    for (flat, &value) in values.iter().enumerate() {
        let (mut rest, mut index, mut shift) = (flat, 0usize, 0u32);
        for (range, &d) in ranges.iter().zip(domain).rev() {
            let length = (range.end - range.start) as usize;
            let coordinate = range.start as usize + rest % length;
            rest /= length;
            index |= coordinate.checked_shl(shift).ok_or(LeafError::Memory)?;
            shift += d.next_power_of_two().trailing_zeros();
        }
        *out.get_mut(index).ok_or(LeafError::Internal {
            rule: "the tile's padded source index is in its column",
        })? = lift(value);
    }
    Ok(())
}
fn full(column: &mut Column, values: &[i64]) -> Result<(), LeafError> {
    let ranges: Vec<_> = column.domain.iter().map(|&d| 0..d).collect();
    scatter(&mut column.values, &column.domain, &ranges, values, |v| v)?;
    column.written = values.len();
    Ok(())
}
fn shared(
    tile: &tessera_compiler::SequenceTile,
    id: tessera_ir::TensorId,
) -> Result<&Share, LeafError> {
    tile.shares
        .iter()
        .find(|s| s.tensor == id)
        .ok_or(LeafError::Internal {
            rule: "the tile contains the graph output defining its source domain",
        })
}

impl ModelWitness {
    /// Assemble all sources a canonical model layout reads, using the existing full CPU pass
    /// and the canonical resident/regenerating tile source. `advice` supplies its exact public
    /// tile partition. All inputs must belong to the same sequence and public model.
    /// Real auxiliaries come from `TileWitness::gadget`; virtual u/key/slacks are recovered
    /// from its canonical lookup keys by their affine decomposition, not committed again.
    ///
    /// # Errors
    /// `Internal` for missing/inconsistent sources, domains, tiles or integer boxes;
    /// `Memory` for failed reservations; `Witness` for a tile source failure.
    pub fn new(
        layout: &LeafLayout<ModelColumn>,
        lowering: &Lowering,
        layers: &LayerMap,
        witness: &(impl crate::stages::WitnessSource + ?Sized),
        source: &(impl WitnessSource + ?Sized),
        advice: &ModelAdviceLayout<'_>,
    ) -> Result<Self, LeafError> {
        let mut sites = Vec::new();
        let mut maximum = None;
        for (s, site) in layout.sites().iter().enumerate() {
            let mut real_layers = Vec::new();
            for (l, gadget) in site.layers.iter().enumerate() {
                let node = &lowering.graph.nodes[owner(layers, site.node, l).0 as usize];
                let mut columns = Vec::new();
                for read in layout.reads(s, l).ok_or(LeafError::Internal {
                    rule: "every model site has its ordered reads",
                })? {
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
                    let domain: Vec<_> = read.axes.iter().map(|&a| request.domain[a]).collect();
                    let len = 1usize
                        .checked_shl(variables(&domain).ok_or(LeafError::Memory)?)
                        .ok_or(LeafError::Memory)?;
                    let fill = if read.term == ModelColumn::ExpKey {
                        let id = request.lookup.table;
                        let spec = lowering.tables.iter().find(|t| t.id == id).ok_or(
                            LeafError::Internal {
                                rule: "the Exp table is in the model lowering",
                            },
                        )?;
                        (1i64
                            .checked_shl(u32::from(spec.log2_rows))
                            .ok_or(LeafError::Memory)?)
                            - 1
                    } else {
                        0
                    };
                    let mut column = Column {
                        name: read.term,
                        domain,
                        values: buffer(len, fill)?,
                        written: 0,
                    };
                    if let Some(id) = tensor(read.term, node) {
                        if lowering.graph.tensors[id.0 as usize].shape != column.domain {
                            return own();
                        }
                        full(
                            &mut column,
                            witness.values(id).ok_or(LeafError::Internal {
                                rule: "the full CPU witness contains every graph source",
                            })?,
                        )?;
                    }
                    columns.push(column);
                }
                if matches!(node.op, tessera_ir::Op::SoftmaxRow { .. }) {
                    // The masked key hides its maximum; the native relation still needs it.
                    let domain = lowering.graph.tensors[node.outputs[1].0 as usize]
                        .shape
                        .clone();
                    let len = 1usize
                        .checked_shl(variables(&domain).ok_or(LeafError::Memory)?)
                        .ok_or(LeafError::Memory)?;
                    columns.push(Column {
                        name: ModelColumn::Auxiliary(tessera_compiler::AuxColumn::Maximum),
                        domain,
                        values: buffer(len, 0)?,
                        written: 0,
                    });
                }
                real_layers.push(columns);
            }
            sites.push(real_layers);
        }
        // Canonical attention shares preserve head and query axes, including empty padded
        // blocks. Concatenating tiles would permute heads whenever there is more than one block.
        for tile in advice.tiles() {
            if matches!(tile.tile, Tile::GreedyMaximum { .. }) {
                let values = source.tile(tile).map_err(LeafError::Witness)?;
                let (id, m) = values.greedy_maximum().ok_or(LeafError::Internal {
                    rule: "runtime maximum advice remains a distinct certificate source",
                })?;
                let n = lowering.graph.tensors[lowering.tokens.0 as usize].shape[0];
                if tile.tile != (Tile::GreedyMaximum { gadget: id.0 })
                    || m.len() != n as usize
                    || maximum.is_some()
                {
                    return own();
                }
                let mut padded = buffer(n.next_power_of_two() as usize, 0)?;
                padded[..m.len()].copy_from_slice(m);
                maximum = Some((id, padded));
                let requests: Vec<_> = values.lookups().collect();
                let [request] = requests.as_slice() else {
                    return own();
                };
                let Some((s, _)) = sites.iter().enumerate().find(|(_, layers)| {
                    layers
                        .iter()
                        .flatten()
                        .any(|c| c.name == ModelColumn::GreedyKey)
                }) else {
                    return own();
                };
                let [columns] = sites[s].as_mut_slice() else {
                    return own();
                };
                let [column] = columns.as_mut_slice() else {
                    return own();
                };
                if layout.sites()[s].layers[0].lookups[0].lookup.table != request.table()
                    || column.values.len() != request.keys().len()
                    || request.values().is_some()
                {
                    return own();
                }
                // This domain is outputs x vocabulary, independently padded on both axes.
                for (out, &key) in column.values.iter_mut().zip(request.keys()) {
                    *out = i64::from(key);
                }
                column.written = elements(&column.domain).ok_or(LeafError::Memory)?;
                continue;
            }
            let id = match tile.tile {
                Tile::Node { node } | Tile::Attention { node, .. } => NodeId(node),
                _ => continue,
            };
            let target = layout.sites().iter().enumerate().find_map(|(s, site)| {
                if sites[s]
                    .iter()
                    .flatten()
                    .any(|c| c.name == ModelColumn::GreedyKey)
                {
                    return None;
                }
                (0..site.layers.len())
                    .find(|&l| owner(layers, site.node, l) == id)
                    .map(|l| (s, l))
            });
            let Some((s, l)) = target else {
                continue;
            };
            let node = &lowering.graph.nodes[id.0 as usize];
            let values = source.tile(tile).map_err(LeafError::Witness)?;
            let requests: Vec<_> = values.lookups().collect();
            let gadget = &layout.sites()[s].layers[l];
            if requests.len() != gadget.lookups.len()
                || !requests
                    .iter()
                    .map(|r| r.table())
                    .eq(gadget.lookups.iter().map(|r| r.lookup.table))
            {
                return own();
            }
            for column in &mut sites[s][l] {
                if tensor(column.name, node).is_some() {
                    continue;
                }
                let graph_output = if column.name == ModelColumn::ExpKey {
                    node.outputs[0]
                } else if matches!(node.op, tessera_ir::Op::SoftmaxRow { .. }) {
                    node.outputs[1]
                } else {
                    node.outputs[0]
                };
                let share = shared(tile, graph_output)?;
                match column.name {
                    ModelColumn::Auxiliary(role) => {
                        let data = values.gadget().iter().find(|(r, _)| *r == role).ok_or(
                            LeafError::Internal {
                                rule: "the canonical tile contains every requested auxiliary role",
                            },
                        )?;
                        scatter(
                            &mut column.values,
                            &column.domain,
                            &share.ranges,
                            &data.1,
                            |v| v,
                        )?;
                        column.written = column
                            .written
                            .checked_add(data.1.len())
                            .ok_or(LeafError::Memory)?;
                    }
                    c if virtual_column(c) => {
                        let index = gadget
                            .lookups
                            .iter()
                            .position(|r| r.lookup.key.terms.iter().any(|(_, t)| *t == c))
                            .ok_or(LeafError::Internal {
                                rule: "each virtual column occurs in a canonical key decomposition",
                            })?;
                        let keys = requests[index].keys();
                        scatter(
                            &mut column.values,
                            &column.domain,
                            &share.ranges,
                            keys,
                            i64::from,
                        )?;
                        column.written = column
                            .written
                            .checked_add(keys.len())
                            .ok_or(LeafError::Memory)?;
                    }
                    _ => return own(),
                }
            }
        }
        for (s, real_layers) in sites.iter_mut().enumerate() {
            for (l, columns) in real_layers.iter_mut().enumerate() {
                if columns
                    .iter()
                    .any(|c| elements(&c.domain) != Some(c.written))
                {
                    return own();
                }
                for index in 0..columns.len() {
                    let name = columns[index].name;
                    if !virtual_column(name) {
                        continue;
                    }
                    let form = &layout.sites()[s].layers[l]
                        .lookups
                        .iter()
                        .find(|r| r.lookup.key.terms.iter().any(|(_, c)| *c == name))
                        .ok_or(LeafError::Internal {
                            rule: "a virtual column has its defining lookup",
                        })?
                        .lookup
                        .key;
                    recover(columns, index, form)?;
                }
            }
        }
        let mut result = Self {
            columns: sites,
            forests: Vec::new(),
            maximum,
        };
        result.build_forests(layout)?;
        Ok(result)
    }

    /// Lend the ordered sources to `prepare_leaf`/`prove_leaf`; no values are copied out.
    pub fn tensors(&self) -> Vec<Vec<SiteTensors<'_, ModelColumn>>> {
        self.columns
            .iter()
            .map(|layers| {
                layers
                    .iter()
                    .map(|columns| {
                        SiteTensors::new(columns.iter().map(|c| (c.name, &c.values[..])).collect())
                    })
                    .collect()
            })
            .collect()
    }
    /// A canonical padded source, lent under this type's borrower rules.
    pub fn values(
        &self,
        site: usize,
        layer: usize,
        column: ModelColumn,
    ) -> Option<&SecretBuf<i64>> {
        self.columns
            .get(site)?
            .get(layer)?
            .iter()
            .find(|c| c.name == column)
            .map(|c| &c.values)
    }
    /// The padded runtime maximum, with its own certificate identity. Values remain secret
    /// and are lent only to the canonical native proof/commitment borrowers.
    pub fn greedy_maximum(&self) -> Option<(tessera_ir::GadgetId, &[i64])> {
        self.maximum.as_ref().map(|(id, m)| (*id, &m[..]))
    }
    /// Product forests in template-node order, with padded layer/head/query roots equal to one.
    /// No forest is sent for a one-key maximum; its scalar root remains a native obligation.
    pub fn forests(&self) -> Vec<Forest<'_, GoldilocksExt2>> {
        self.forests
            .iter()
            .map(|f| Forest::new(&f.leaves, f.height))
            .collect()
    }
    /// The product forests as a grouped stage splits them (RFC-0054 'The forest'), in the order of
    /// [`forests`](Self::forests): each leaf table indexed by layer, head, query and key, the
    /// queries from the keys' height up and the layers above the heads.
    pub fn stage_forests(&self) -> Vec<StageForest<'_>> {
        self.forests
            .iter()
            .map(|f| StageForest::new(&f.leaves, f.height as u32, f.height as u32, f.layers))
            .collect()
    }
    /// The public sites of the product forests, in the same order as `forests`.
    pub fn forest_sites(&self) -> impl Iterator<Item = NodeRef> + '_ {
        self.forests.iter().map(|f| f.site)
    }
    fn build_forests(&mut self, layout: &LeafLayout<ModelColumn>) -> Result<(), LeafError> {
        for (s, site) in layout.sites().iter().enumerate() {
            let Some(column) = self.columns[s][0]
                .iter()
                .find(|c| c.name == ModelColumn::ExpKey)
            else {
                continue;
            };
            let &[heads, n, keys] = column.domain.as_slice() else {
                return own();
            };
            if keys != n {
                return own();
            }
            let height = n.next_power_of_two().trailing_zeros();
            if height == 0 {
                continue;
            }
            let block = column.values.len();
            let count = block
                .checked_mul(site.layers.len().next_power_of_two())
                .ok_or(LeafError::Memory)?;
            let mut leaves = buffer(count, Goldilocks::ONE)?;
            for l in 0..site.layers.len() {
                let values = self
                    .values(s, l, ModelColumn::ExpKey)
                    .ok_or(LeafError::Internal {
                        rule: "each Softmax layer has its masked key column",
                    })?;
                for head in 0..heads {
                    for query in 0..n {
                        for key in 0..=query {
                            let index = ((head as usize) << (2 * height))
                                | ((query as usize) << height)
                                | key as usize;
                            leaves[l * block + index] = Goldilocks::from_i64(values[index]);
                        }
                    }
                }
            }
            self.forests.push(Trees {
                site: site.node,
                height: height as usize,
                layers: block.trailing_zeros(),
                leaves,
            });
        }
        Ok(())
    }
}

/// Invert the one affine key that defines a virtual column from its range/Exp lookup.
/// This constructs a witness only: the driver still owes every native source relation.
fn recover(
    columns: &mut [Column],
    index: usize,
    form: &Affine<Goldilocks, ModelColumn>,
) -> Result<(), LeafError> {
    let name = columns[index].name;
    let mut coefficient = Goldilocks::ZERO;
    for &(a, c) in &form.terms {
        if c == name {
            coefficient += a;
        }
    }
    let inverse = coefficient.inverse().ok_or(LeafError::Internal {
        rule: "a virtual column has an invertible affine coefficient",
    })?;
    let mut fits = true;
    for row in 0..columns[index].values.len() {
        let mut value = Goldilocks::from_i64(columns[index].values[row]) - form.constant;
        for &(a, c) in &form.terms {
            if c == name {
                continue;
            }
            let source = columns
                .iter()
                .find(|t| t.name == c)
                .ok_or(LeafError::Internal {
                    rule: "a virtual key's operands occur in the split",
                })?;
            if source.domain != columns[index].domain {
                return own();
            }
            value -= a * Goldilocks::from_i64(source.values[row]);
        }
        let integer = (value * inverse).to_centered_i128();
        let valid = (integer as u128) < (1u128 << 62);
        fits &= valid;
        columns[index].values[row] = core::hint::select_unpredictable(valid, integer as i64, 0);
    }
    if !fits {
        return own();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_compiler::AuxColumn;
    fn column(name: ModelColumn, domain: &[u32], values: &[i64]) -> Column {
        let mut buffer = buffer(values.len(), 0).unwrap();
        buffer.copy_from_slice(values);
        Column {
            name,
            domain: domain.to_vec(),
            values: buffer,
            written: values.len(),
        }
    }
    #[test]
    fn affine_recovery_checks_its_integer_box_and_own_operands() {
        let x = ModelColumn::RootInput;
        let low = ModelColumn::Auxiliary(AuxColumn::InputLimb(0));
        let inv = Goldilocks::from_u64_reduced(16).inverse().unwrap();
        let form = Affine {
            constant: Goldilocks::ZERO,
            terms: vec![(inv, x), (-inv, low)],
        };
        let mut columns = vec![column(x, &[2], &[2, 3]), column(low, &[2], &[11, 12])];
        recover(&mut columns, 0, &form).unwrap();
        assert_eq!(&columns[0].values[..], &[43, 60]);
        let mut columns = vec![column(x, &[2], &[2, 3]), column(low, &[2], &[11, 12])];
        let mut affine = form.clone();
        affine.constant = Goldilocks::from_i64(-5);
        recover(&mut columns, 0, &affine).unwrap();
        assert_eq!(&columns[0].values[..], &[123, 140]);

        for value in [-1, 1 << 62] {
            let mut columns = vec![column(x, &[], &[value])];
            assert!(matches!(
                recover(
                    &mut columns,
                    0,
                    &Affine {
                        constant: Goldilocks::ZERO,
                        terms: vec![(Goldilocks::ONE, x)]
                    }
                ),
                Err(LeafError::Internal {
                    rule: "model leaf columns cover every canonical source at its public coordinates"
                })
            ));
        }
        let mut columns = vec![column(x, &[], &[0])];
        assert!(matches!(
            recover(
                &mut columns,
                0,
                &Affine {
                    constant: Goldilocks::ZERO,
                    terms: vec![(Goldilocks::ZERO, x)]
                }
            ),
            Err(LeafError::Internal {
                rule: "a virtual column has an invertible affine coefficient"
            })
        ));
        let mut columns = vec![column(x, &[2], &[2, 3]), column(low, &[1], &[11])];
        assert!(matches!(
            recover(&mut columns, 0, &form),
            Err(LeafError::Internal {
                rule: "model leaf columns cover every canonical source at its public coordinates"
            })
        ));
        let mut columns = vec![column(x, &[2], &[2, 3])];
        assert!(matches!(
            recover(&mut columns, 0, &form),
            Err(LeafError::Internal {
                rule: "a virtual key's operands occur in the split"
            })
        ));
    }
    #[test]
    fn scatter_validates_public_ranges_lengths_and_empty_or_scalar_domains() {
        let mut out = vec![0; 8];
        for ranges in [
            vec![core::ops::Range { start: 0, end: 2 }],
            vec![0..3, 0..2],
            vec![core::ops::Range { start: 1, end: 0 }, 0..2],
        ] {
            assert!(matches!(
                scatter(&mut out, &[2, 4], &ranges, &[1, 2], |v| v),
                Err(LeafError::Internal { .. })
            ));
            assert_eq!(out, vec![0; 8]);
        }
        // The extra row fits the padded allocation, but exceeds the real head domain.
        assert!(matches!(
            scatter(&mut out, &[3, 2], &[0..4, 0..2], &[7; 8], |v| v),
            Err(LeafError::Internal { .. })
        ));
        assert_eq!(out, vec![0; 8]);
        assert!(matches!(
            scatter(&mut out, &[2, 4], &[0..2, 0..4], &[1, 2], |v| v),
            Err(LeafError::Internal { .. })
        ));
        assert!(matches!(
            scatter(&mut [], &[1], core::slice::from_ref(&(0..1)), &[7], |v| v),
            Err(LeafError::Internal { .. })
        ));
        scatter(&mut out, &[2, 4], &[0..2, 4..4], &[] as &[i64], |v| v).unwrap();
        let mut scalar = [0];
        scatter(&mut scalar, &[], &[], &[7], |v| v).unwrap();
        assert_eq!(scalar, [7]);
    }
}
