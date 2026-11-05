// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical source scopes; no production memory admission is implied (RFC-0077).

use super::*;
use crate::leaf::grouped::{ForestShape, GroupColumns, GroupInputs};
use crate::leaf::{GroupLayout, GroupedLayout};
use tessera_field::Field;
use tessera_logup::Tuples;

#[cfg(test)]
mod proof_tests;

#[derive(Clone, Copy)]
pub(super) enum Selection<'a> {
    All,
    Group(&'a GroupedLayout, &'a GroupLayout),
    Read {
        site: usize,
        layer: usize,
        term: ModelColumn,
    },
    Forest {
        site: usize,
    },
}

impl Selection<'_> {
    pub(super) fn projection(
        self,
        layout: &LeafLayout<ModelColumn>,
        s: usize,
        l: usize,
        term: ModelColumn,
        vars: u32,
    ) -> Result<Option<Projection>, LeafError> {
        match self {
            Self::All => Projection::full(vars).map(Some),
            Self::Group(grouped, group) => {
                let reads = layout.reads(s, l).ok_or(LeafError::Internal {
                    rule: "a scoped source has its canonical ordered reads",
                })?;
                let Some(read) = reads.iter().position(|r| r.term == term) else {
                    return Ok(None);
                };
                let Some(read) = group
                    .reads()
                    .iter()
                    .find(|r| r.site == s && r.layer == l && r.read == read)
                else {
                    return Ok(None);
                };
                let polynomial =
                    grouped
                        .polynomials()
                        .get(read.polynomial)
                        .ok_or(LeafError::Internal {
                            rule: "a group source names its original polynomial",
                        })?;
                if polynomial.vars != vars as usize {
                    return own();
                }
                Projection::new(vars, &polynomial.group_vars, read.group_bits).map(Some)
            }
            Self::Read {
                site,
                layer,
                term: target,
            } => {
                if (site, layer) != (s, l) {
                    return Ok(None);
                }
                let defining = layout.sites()[s].layers[l]
                    .lookups
                    .iter()
                    .find(|r| r.lookup.key.terms.iter().any(|(_, c)| *c == target));
                let needed = target == term
                    || virtual_column(target)
                        && defining
                            .is_some_and(|r| r.lookup.key.terms.iter().any(|(_, c)| *c == term));
                if needed {
                    Projection::full(vars).map(Some)
                } else {
                    Ok(None)
                }
            }
            Self::Forest { site } => {
                if s == site && term == ModelColumn::ExpKey {
                    Projection::full(vars).map(Some)
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// A factory borrows the real graph/tile source; its backing/replay work stays additional.
pub(crate) struct ModelSourceFactory<'a> {
    pub(crate) layout: &'a LeafLayout<ModelColumn>,
    pub(crate) grouped: &'a GroupedLayout,
    pub(crate) lowering: &'a Lowering,
    pub(crate) layers: &'a LayerMap,
    pub(crate) witness: &'a dyn crate::stages::WitnessSource,
    pub(crate) source: &'a dyn WitnessSource,
    pub(crate) advice: &'a ModelAdviceLayout<'a>,
}

/// Owns only selected columns and forest shares; cannot outlive its source factory.
pub(crate) struct GroupSources<'a> {
    pub(crate) layout: GroupLayout,
    pub(super) inner: ModelWitness,
    factory: &'a ModelSourceFactory<'a>,
}

/// Original native inputs either borrow the resident oracle or own clearing regeneration.
pub(crate) enum ColumnLease<'a> {
    Borrowed(&'a [i64]),
    Owned(SecretBuf<i64>),
}

impl core::ops::Deref for ColumnLease<'_> {
    type Target = [i64];
    fn deref(&self) -> &[i64] {
        match self {
            Self::Borrowed(v) => v,
            Self::Owned(v) => v,
        }
    }
}

pub(crate) trait ModelLeafSource {
    fn column(
        &self,
        site: usize,
        layer: usize,
        term: ModelColumn,
    ) -> Result<ColumnLease<'_>, LeafError>;
    fn maximum(&self) -> Result<(tessera_ir::GadgetId, ColumnLease<'_>), LeafError>;
}

impl ModelLeafSource for ModelWitness {
    fn column(
        &self,
        site: usize,
        layer: usize,
        term: ModelColumn,
    ) -> Result<ColumnLease<'_>, LeafError> {
        self.values(site, layer, term)
            .map(|v| ColumnLease::Borrowed(v))
            .ok_or(LeafError::Internal {
                rule: "a native source has its canonical model column",
            })
    }
    fn maximum(&self) -> Result<(tessera_ir::GadgetId, ColumnLease<'_>), LeafError> {
        self.greedy_maximum()
            .map(|(id, v)| (id, ColumnLease::Borrowed(v)))
            .ok_or(LeafError::Internal {
                rule: "a native source has its canonical maximum certificate",
            })
    }
}

impl ModelLeafSource for ModelSourceFactory<'_> {
    fn column(
        &self,
        site: usize,
        layer: usize,
        term: ModelColumn,
    ) -> Result<ColumnLease<'_>, LeafError> {
        let mut source = ModelWitness::build(
            self.layout,
            self.lowering,
            self.layers,
            self.witness,
            self.source,
            self.advice,
            Selection::Read { site, layer, term },
        )?;
        let columns = source
            .columns
            .get_mut(site)
            .and_then(|s| s.get_mut(layer))
            .ok_or(LeafError::Internal {
                rule: "a native source has its canonical site and layer",
            })?;
        let index = columns
            .iter()
            .position(|c| c.name == term)
            .ok_or(LeafError::Internal {
                rule: "a native source has its canonical model column",
            })?;
        Ok(ColumnLease::Owned(columns.remove(index).values))
    }

    fn maximum(&self) -> Result<(tessera_ir::GadgetId, ColumnLease<'_>), LeafError> {
        let mut tiles = self
            .advice
            .tiles()
            .iter()
            .filter(|t| matches!(t.tile, Tile::GreedyMaximum { .. }));
        let (Some(tile), None) = (tiles.next(), tiles.next()) else {
            return own();
        };
        let values = self.source.tile(tile).map_err(LeafError::Witness)?;
        let (id, maximum) = values.greedy_maximum().ok_or(LeafError::Internal {
            rule: "a native source has its canonical maximum certificate",
        })?;
        let n = self.lowering.graph.tensors[self.lowering.tokens.0 as usize].shape[0];
        if tile.tile != (Tile::GreedyMaximum { gadget: id.0 }) || maximum.len() != n as usize {
            return own();
        }
        let mut padded = buffer(
            n.checked_next_power_of_two().ok_or(LeafError::Memory)? as usize,
            0,
        )?;
        padded[..maximum.len()].copy_from_slice(maximum);
        Ok((id, ColumnLease::Owned(padded)))
    }
}

impl<'a> ModelSourceFactory<'a> {
    pub(in crate::leaf) fn forest_shapes(&self) -> Result<Vec<ForestShape>, LeafError> {
        let count = (0..self.layout.sites().len()).try_fold(0usize, |n, s| {
            n.checked_add(usize::from(forest_domain(self.layout, s)?.is_some()))
                .ok_or(LeafError::Memory)
        })?;
        let mut out = descriptors(count)?;
        for (s, site) in self.layout.sites().iter().enumerate() {
            let Some(domain) = forest_domain(self.layout, s)? else {
                continue;
            };
            let height = domain[2]
                .checked_next_power_of_two()
                .ok_or(LeafError::Memory)?
                .trailing_zeros();
            let layers = variables(&domain).ok_or(LeafError::Memory)?;
            let layer_bits = site
                .layers
                .len()
                .checked_next_power_of_two()
                .ok_or(LeafError::Memory)?
                .trailing_zeros();
            out.push(ForestShape {
                vars: layers.checked_add(layer_bits).ok_or(LeafError::Memory)?,
                height,
                queries: height,
                layers,
            });
        }
        Ok(out)
    }

    pub(in crate::leaf) fn group_inputs<'r>(
        &'a self,
        h: usize,
        rows: impl Fn(tessera_tables::TableId) -> Result<Tuples<'r, Goldilocks>, LeafError>,
    ) -> Result<GroupInputs<'a, ModelColumn>, LeafError> {
        let group = self.group(h)?;
        let columns = group.columns(h, rows)?;
        let mut splits = descriptors(group.layout.reads().len())?;
        for read in group.layout.reads() {
            let source = group.tensor(read.site, read.layer, read.read)?;
            let mut values = buffer(source.len(), Goldilocks::ZERO)?;
            for (out, &value) in values.iter_mut().zip(source) {
                *out = Goldilocks::from_i64(value);
            }
            splits.push(values);
        }
        let mut shares = descriptors(group.inner.forests.len())?;
        for forest in group.inner.forests {
            shares.push(forest.leaves);
        }
        GroupInputs::scoped(h, group.layout, columns, shares, splits)
    }

    pub(crate) fn group(&'a self, h: usize) -> Result<GroupSources<'a>, LeafError> {
        let layout = self.grouped.group(h)?;
        let inner = ModelWitness::build(
            self.layout,
            self.lowering,
            self.layers,
            self.witness,
            self.source,
            self.advice,
            Selection::Group(self.grouped, &layout),
        )?;
        Ok(GroupSources {
            layout,
            inner,
            factory: self,
        })
    }

    /// Original tensors followed by original forest leaves, in the canonical merge order.
    pub(crate) fn original(&self, polynomial: usize) -> Result<SecretBuf<Goldilocks>, LeafError> {
        let mut index = 0;
        for (s, site) in self.layout.sites().iter().enumerate() {
            for l in 0..site.layers.len() {
                let reads = self.layout.reads(s, l).ok_or(LeafError::Internal {
                    rule: "original model polynomials have their canonical reads",
                })?;
                for read in 0..reads.len() {
                    if index == polynomial {
                        return self.original_tensor(s, l, read);
                    }
                    index += 1;
                }
            }
        }
        if index != self.grouped.polynomials().len() {
            return own();
        }
        for (s, site) in self.layout.sites().iter().enumerate() {
            if forest_domain(self.layout, s)?.is_none() {
                continue;
            }
            if index == polynomial {
                let mut source = ModelWitness::build(
                    self.layout,
                    self.lowering,
                    self.layers,
                    self.witness,
                    self.source,
                    self.advice,
                    Selection::Forest { site: s },
                )?;
                if source.forests.len() != 1 || source.forests[0].site != site.node {
                    return own();
                }
                return Ok(source.forests.remove(0).leaves);
            }
            index += 1;
        }
        own()
    }

    /// Regenerate one tensor in full natural order; no group-local permutation is returned.
    pub(crate) fn original_tensor(
        &self,
        s: usize,
        l: usize,
        read: usize,
    ) -> Result<SecretBuf<Goldilocks>, LeafError> {
        let term = self
            .layout
            .reads(s, l)
            .and_then(|r| r.get(read))
            .ok_or(LeafError::Internal {
                rule: "an original source names a canonical model read",
            })?
            .term;
        let values = self.column(s, l, term)?;
        let mut out = buffer(values.len(), Goldilocks::ZERO)?;
        for (out, &value) in out.iter_mut().zip(values.iter()) {
            *out = Goldilocks::from_i64(value);
        }
        Ok(out)
    }
}

impl GroupSources<'_> {
    fn columns<'r>(
        &self,
        h: usize,
        rows: impl Fn(tessera_tables::TableId) -> Result<Tuples<'r, Goldilocks>, LeafError>,
    ) -> Result<GroupColumns, LeafError> {
        let stage = self.factory.layout;
        let mut out = descriptors(self.layout.instances().len())?;
        for instance in self.layout.instances() {
            let rows = rows(instance.table)?;
            let key_0 = *rows.keys.first().ok_or(LeafError::Internal {
                rule: "an instance's table has a row",
            })?;
            let value_0 = rows.values.and_then(|v| v.first()).copied();
            let leaves = 1usize
                .checked_shl(instance.height)
                .ok_or(LeafError::Memory)?;
            let mut keys = buffer(leaves, key_0)?;
            let mut values = value_0.map(|v| buffer(leaves, v)).transpose()?;
            for block in &instance.blocks {
                let request = stage
                    .sites()
                    .get(block.site)
                    .and_then(|s| s.layers.get(block.layer))
                    .and_then(|g| g.lookups.get(block.lookup))
                    .ok_or(LeafError::Internal {
                        rule: "a scoped block has its canonical lookup identity",
                    })?;
                if request.lookup.value.is_some() != values.is_some() {
                    return own();
                }
                let at = |term: ModelColumn, x: u64| -> Result<Goldilocks, LeafError> {
                    let read = stage
                        .reads(block.site, block.layer)
                        .and_then(|r| r.iter().find(|r| r.term == term))
                        .ok_or(LeafError::Internal {
                            rule: "a scoped lookup term has its canonical read",
                        })?;
                    let mut original = 0usize;
                    for (j, &coordinate) in read.coords.iter().enumerate() {
                        let bit = x.checked_shr(coordinate).ok_or(LeafError::Memory)? & 1;
                        original |= (bit as usize)
                            .checked_shl(u32::try_from(j).map_err(|_| LeafError::Memory)?)
                            .ok_or(LeafError::Memory)?;
                    }
                    let column = self.inner.columns[block.site][block.layer]
                        .iter()
                        .find(|c| c.name == term)
                        .ok_or(LeafError::Internal {
                            rule: "a scoped lookup has its selected operand",
                        })?;
                    let selected =
                        column
                            .projection
                            .selected(original)
                            .ok_or(LeafError::Internal {
                                rule: "a scoped lookup and its source fix the same coordinates",
                            })?;
                    let value = column.values.get(selected).ok_or(LeafError::Internal {
                        rule: "a selected lookup index fits its source",
                    })?;
                    Ok(Goldilocks::from_i64(*value))
                };
                let form = |form: &Affine<Goldilocks, ModelColumn>, x| {
                    form.terms
                        .iter()
                        .try_fold(form.constant, |sum, &(a, term)| {
                            Ok::<_, LeafError>(sum + a * at(term, x)?)
                        })
                };
                for j in 0..block.entries() {
                    let x =
                        self.factory
                            .grouped
                            .source(h, block, j)
                            .ok_or(LeafError::Internal {
                                rule: "a block's entries are its stage block's",
                            })?;
                    let leaf = crate::leaf::leaf_index(instance.height, block.offset + j) as usize;
                    keys[leaf] = form(&request.lookup.key, x)?;
                    if let (Some(values), Some(value)) = (&mut values, &request.lookup.value) {
                        values[leaf] = form(value, x)?;
                    }
                }
            }
            out.push((instance.table, keys, values));
        }
        Ok(out)
    }

    pub(crate) fn tensor(
        &self,
        site: usize,
        layer: usize,
        read: usize,
    ) -> Result<&[i64], LeafError> {
        let term = self
            .factory
            .layout
            .reads(site, layer)
            .and_then(|r| r.get(read))
            .ok_or(LeafError::Internal {
                rule: "a selected source names a canonical model read",
            })?
            .term;
        self.inner
            .values(site, layer, term)
            .map(|v| &v[..])
            .ok_or(LeafError::Internal {
                rule: "a selected source belongs to its group",
            })
    }

    pub(crate) fn forests(&self) -> impl Iterator<Item = Forest<'_, GoldilocksExt2>> {
        self.inner
            .forests
            .iter()
            .map(|f| Forest::new(&f.leaves, f.height))
    }
}

fn forest_domain(
    layout: &LeafLayout<ModelColumn>,
    s: usize,
) -> Result<Option<[u32; 3]>, LeafError> {
    let site = layout.sites().get(s).ok_or(LeafError::Internal {
        rule: "a forest has its model site",
    })?;
    let reads = layout.reads(s, 0).ok_or(LeafError::Internal {
        rule: "a forest site has real layers",
    })?;
    let Some(read) = reads.iter().find(|r| r.term == ModelColumn::ExpKey) else {
        return Ok(None);
    };
    let request = site.layers[0]
        .lookups
        .iter()
        .find(|r| {
            r.lookup
                .key
                .terms
                .iter()
                .any(|(_, c)| *c == ModelColumn::ExpKey)
        })
        .ok_or(LeafError::Internal {
            rule: "a forest's masked key has its canonical lookup",
        })?;
    let domain: Vec<_> = read
        .axes
        .iter()
        .map(|&a| request.domain.get(a).copied())
        .collect();
    let [Some(heads), Some(queries), Some(keys)] = domain.as_slice() else {
        return own();
    };
    if *keys <= 1 {
        return Ok(None);
    }
    Ok(Some([*heads, *queries, *keys]))
}

impl ModelWitness {
    pub(super) fn build_group_forests(
        &mut self,
        layout: &LeafLayout<ModelColumn>,
        lowering: &Lowering,
        grouped: &GroupedLayout,
        group: &GroupLayout,
    ) -> Result<(), LeafError> {
        let mut forests = 0usize;
        for s in 0..layout.sites().len() {
            forests += usize::from(forest_domain(layout, s)?.is_some());
        }
        self.forests = descriptors(forests)?;
        let g_t = u32::from(grouped.groups().token_bits);
        let g_l = u32::from(grouped.groups().layer_bits);
        for (s, site) in layout.sites().iter().enumerate() {
            let Some(domain @ [heads, n, keys]) = forest_domain(layout, s)? else {
                continue;
            };
            let height = keys
                .checked_next_power_of_two()
                .ok_or(LeafError::Memory)?
                .trailing_zeros();
            let query_bits = n
                .checked_next_power_of_two()
                .ok_or(LeafError::Memory)?
                .trailing_zeros();
            let block_bits = variables(&domain).ok_or(LeafError::Memory)?;
            let layer_bits = site
                .layers
                .len()
                .checked_next_power_of_two()
                .ok_or(LeafError::Memory)?
                .trailing_zeros();
            if g_t > query_bits || g_l > layer_bits {
                return own();
            }
            let vars = block_bits
                .checked_add(layer_bits)
                .ok_or(LeafError::Memory)?;
            let mut positions = descriptors((g_t + g_l) as usize)?;
            positions.extend(
                (height..height + g_t)
                    .chain(block_bits..block_bits + g_l)
                    .map(|v| v as usize),
            );
            let bits = group.tokens() | (group.layers() << g_t);
            let projection = Projection::new(vars, &positions, bits)?;
            let mut leaves = buffer(projection.len(), Goldilocks::ONE)?;
            let local_mask = (1usize << block_bits) - 1;
            let key_mask = (1usize << height) - 1;
            let query_mask = (1usize << query_bits) - 1;
            for (selected, value) in leaves.iter_mut().enumerate() {
                let original = projection.original(selected).ok_or(LeafError::Memory)?;
                let layer = original >> block_bits;
                let local = original & local_mask;
                let head = local >> (height + query_bits);
                let query = (local >> height) & query_mask;
                let key = local & key_mask;
                if layer >= site.layers.len() || head >= heads as usize || query >= n as usize {
                    continue;
                }
                let position = match &lowering.graph.slots {
                    Some(slots) => {
                        let ordinal = query / slots.slot_len as usize;
                        let position = query % slots.slot_len as usize;
                        if slots
                            .lengths
                            .get(ordinal)
                            .is_none_or(|&n| position >= n as usize)
                        {
                            continue;
                        }
                        position
                    }
                    None => query,
                };
                if key > position {
                    continue;
                }
                let source = self.columns[s][layer]
                    .iter()
                    .find(|c| c.name == ModelColumn::ExpKey)
                    .ok_or(LeafError::Internal {
                        rule: "a selected forest has its scoped masked-key source",
                    })?;
                let index = source
                    .projection
                    .selected(local)
                    .ok_or(LeafError::Internal {
                        rule: "a forest and its source fix identical query coordinates",
                    })?;
                *value =
                    Goldilocks::from_i64(*source.values.get(index).ok_or(LeafError::Internal {
                        rule: "a selected forest index belongs to its masked key",
                    })?);
            }
            self.forests.push(Trees {
                site: site.node,
                height: height as usize,
                layers: block_bits - g_t,
                leaves,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_codec::{Ascii, DecodeLimits, Secret, envelope, schema};
    use tessera_compiler::{PublicModel, analyze, claim_list, select_model};
    use tessera_hash::ParamSetId;
    use tessera_ir::{GadgetPolicy, IntegerModel};
    use tessera_protocol::{LeafGroups, ParamSet};
    use tessera_witness::{Engine, Resident};

    fn selected(index: usize, positions: &[usize], bits: u64) -> bool {
        positions
            .iter()
            .enumerate()
            .all(|(j, &v)| (index >> v) & 1 == ((bits >> j) & 1) as usize)
    }

    #[test]
    fn canonical_group_and_original_sources_match_full_model_columns_and_forests() {
        let mut cases = 0;
        let mut smaller = 0;
        for name in [
            "tiny-2-minimal",
            "tiny-2-servable",
            "tiny-2-affine",
            "tiny-4-servable",
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../spec/vectors/models")
                .join(format!("{name}.d"));
            let model: IntegerModel = envelope::open(
                &std::fs::read(path).unwrap(),
                &schema::INTEGER_MODEL,
                DecodeLimits::INTEGER_MODEL,
            )
            .unwrap();
            for n in [3, 5] {
                let lowering = tessera_compiler::lower(PublicModel::of(&model), n).unwrap();
                let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
                let policy = GadgetPolicy::default();
                let selection =
                    select_model(tessera_ir::PROTOCOL_VERSION, &lowering, &ranges, &policy)
                        .unwrap();
                let p = ParamSet::TEST_SMALL;
                let claims = claim_list(
                    PublicModel::of(&model),
                    &selection,
                    &policy,
                    p.layout,
                    ParamSetId(Ascii::new(p.id).unwrap()),
                )
                .unwrap();
                let layers = LayerMap::new(&lowering).unwrap();
                let layout = LeafLayout::model(&lowering, &layers, &selection, &claims).unwrap();
                let advice = ModelAdviceLayout::new(&lowering, &selection, &claims).unwrap();
                let engine = Engine::new(&lowering).unwrap();
                let tokens = Secret::new((0..n).map(|i| (i + 1) % model.arch.vocab).collect());
                let witness = engine.run(&model, &tokens).unwrap();
                let source = Resident::new(&engine, &selection, &model, &witness).unwrap();
                let full =
                    ModelWitness::new(&layout, &lowering, &layers, &witness, &source, &advice)
                        .unwrap();
                for (gt, gl) in [(0, 1), (1, 0), (1, 1), (2, 1)] {
                    let grouped = GroupedLayout::new(
                        &layout,
                        &layers,
                        LeafGroups {
                            token_bits: gt,
                            layer_bits: gl,
                        },
                    )
                    .unwrap();
                    let factory = ModelSourceFactory {
                        layout: &layout,
                        grouped: &grouped,
                        lowering: &lowering,
                        layers: &layers,
                        witness: &witness,
                        source: &source,
                        advice: &advice,
                    };
                    let full_cells: usize = full
                        .columns
                        .iter()
                        .flatten()
                        .flatten()
                        .map(|c| c.values.len())
                        .sum();
                    for h in 0..grouped.count() {
                        let group = factory.group(h).unwrap();
                        cases += 1;
                        let mut cells = 0;
                        for read in group.layout.reads() {
                            let term = layout.reads(read.site, read.layer).unwrap()[read.read].term;
                            let original = full.values(read.site, read.layer, term).unwrap();
                            let polynomial = &grouped.polynomials()[read.polynomial];
                            let expected: Vec<_> = original
                                .iter()
                                .copied()
                                .enumerate()
                                .filter(|(i, _)| {
                                    selected(*i, &polynomial.group_vars, read.group_bits)
                                })
                                .map(|(_, v)| v)
                                .collect();
                            assert_eq!(
                                group.tensor(read.site, read.layer, read.read).unwrap(),
                                expected,
                                "{name} n{n} ({gt},{gl}) group{h} read{}",
                                read.polynomial
                            );
                            cells += expected.len();
                        }
                        smaller += usize::from(cells < full_cells);
                        assert_eq!(group.inner.forests.len(), full.forests.len());
                        assert_eq!(group.forests().count(), full.forests.len());
                        for (original, actual) in full.forests.iter().zip(&group.inner.forests) {
                            let positions: Vec<_> = (original.height
                                ..original.height + gt as usize)
                                .chain(
                                    original.layers as usize
                                        ..original.layers as usize + gl as usize,
                                )
                                .collect();
                            let expected: Vec<_> = original
                                .leaves
                                .iter()
                                .copied()
                                .enumerate()
                                .filter(|(i, _)| selected(*i, &positions, h as u64))
                                .map(|(_, v)| v)
                                .collect();
                            assert_eq!(actual.site, original.site);
                            assert_eq!(
                                &actual.leaves[..],
                                expected,
                                "{name} forest ({gt},{gl}) group{h}"
                            );
                        }
                    }
                    let mut index = 0;
                    for (s, site) in layout.sites().iter().enumerate() {
                        for l in 0..site.layers.len() {
                            for read in layout.reads(s, l).unwrap() {
                                let expected: Vec<_> = full
                                    .values(s, l, read.term)
                                    .unwrap()
                                    .iter()
                                    .map(|&v| Goldilocks::from_i64(v))
                                    .collect();
                                assert_eq!(
                                    &factory.original(index).unwrap()[..],
                                    expected,
                                    "{name} original {index}"
                                );
                                index += 1;
                            }
                        }
                    }
                    for forest in &full.forests {
                        assert_eq!(&factory.original(index).unwrap()[..], &forest.leaves[..]);
                        index += 1;
                    }
                    assert!(matches!(
                        factory.original(index),
                        Err(LeafError::Internal { .. })
                    ));
                    assert!(matches!(
                        factory.group(grouped.count()),
                        Err(LeafError::Internal { .. })
                    ));
                }
            }
        }
        assert!(cases >= 100 && smaller >= 100);
    }
}
