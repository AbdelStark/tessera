// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Candidate index insertion and stateless group source, separate from the oracle.

use serde::Serialize;
use tessera_codec::SecretBuf;
use tessera_field::{Goldilocks as G, PrimeField};
use tessera_prover::leaf::{GroupLayout, GroupedLayout, LeafError};
use tessera_tables::TableId;

use crate::fixture::{Fixture, ForestGeometry, bits};

pub type Columns = Vec<(TableId, SecretBuf<G>, Option<SecretBuf<G>>)>;

#[derive(Serialize)]
pub struct IndexRecord {
    pub group: u64,
    pub table: u32,
    pub site: usize,
    pub layer: usize,
    pub lookup: usize,
    pub position: u64,
    pub original: u64,
    pub leaf: u64,
}

#[derive(Serialize)]
pub struct ForestRecord {
    pub group: u64,
    pub position: u64,
    pub original: u64,
}

pub struct GroupOutput {
    pub columns: Columns,
    pub forest: SecretBuf<G>,
    pub indices: Vec<IndexRecord>,
    pub forest_indices: Vec<ForestRecord>,
}

/// Insertion by coordinate enumeration, rather than the oracle's word-mask formula.
pub fn insert(index: u64, vars: u32, fixed: impl Fn(u32) -> Option<u64>) -> u64 {
    let mut source = 0;
    let mut free = 0;
    for bit in 0..vars {
        let value = match fixed(bit) {
            Some(value) => value,
            None => {
                let value = (index >> free) & 1;
                free += 1;
                value
            }
        };
        source |= value << bit;
    }
    assert_eq!(index >> free, 0, "share index is in its public domain");
    source
}

pub fn leaf(height: u32, index: u64) -> u64 {
    let mut out = index & 1;
    for bit in 1..height {
        out |= ((index >> bit) & 1) << (height - bit);
    }
    out
}

pub fn group(
    f: &Fixture,
    grouped: &GroupedLayout,
    h: usize,
    layout: &GroupLayout,
    forest: &ForestGeometry,
) -> Result<GroupOutput, LeafError> {
    let gt = u32::from(grouped.groups().token_bits);
    let gl = u32::from(grouped.groups().layer_bits);
    let token = h as u64 & ((1 << gt) - 1);
    let layer = h as u64 >> gt;
    let count: usize = layout
        .instances()
        .iter()
        .flat_map(|i| &i.blocks)
        .map(|b| b.entries() as usize)
        .sum();
    let mut indices = Vec::with_capacity(count);
    let mut columns = Vec::with_capacity(layout.instances().len());
    for instance in layout.instances() {
        let n = 1usize << instance.height;
        let mut keys = SecretBuf::try_with_capacity(n).map_err(|_| LeafError::Memory)?;
        keys.resize(n, G::from_u64_reduced(0));
        let mut values = if instance.table == TableId(2) {
            let mut v = SecretBuf::try_with_capacity(n).map_err(|_| LeafError::Memory)?;
            v.resize(n, G::from_u64_reduced(3));
            Some(v)
        } else {
            None
        };
        for block in &instance.blocks {
            let lookup = &f.stage.sites()[block.site].layers[block.layer].lookups[block.lookup];
            let start = lookup.token.map(|axis| {
                lookup.domain[axis + 1..]
                    .iter()
                    .map(|&n| bits(n))
                    .sum::<u32>()
            });
            let vars = lookup.vars();
            for j in 0..block.entries() {
                let original = insert(j, vars, |bit| {
                    start
                        .filter(|&s| bit >= s && bit < s + gt)
                        .map(|s| (token >> (bit - s)) & 1)
                });
                let at = |term| {
                    let read = f.read(block.site, block.layer, term);
                    let index = read
                        .coords
                        .iter()
                        .enumerate()
                        .fold(0, |x, (i, &bit)| x | (((original >> bit) & 1) << i));
                    G::from_i64(f.datum(block.site, block.layer, term, index))
                };
                let output = leaf(instance.height, block.offset + j);
                keys[output as usize] = lookup.lookup.key.evaluate::<G>(at);
                if let (Some(column), Some(value)) = (&mut values, &lookup.lookup.value) {
                    column[output as usize] = value.evaluate::<G>(at);
                }
                indices.push(IndexRecord {
                    group: h as u64,
                    table: instance.table.0,
                    site: block.site,
                    layer: block.layer,
                    lookup: block.lookup,
                    position: j,
                    original,
                    leaf: output,
                });
            }
        }
        columns.push((instance.table, keys, values));
    }
    let n = 1usize << (forest.vars - gt - gl);
    let mut share = SecretBuf::try_with_capacity(n).map_err(|_| LeafError::Memory)?;
    let mut forest_indices = Vec::with_capacity(n);
    for j in 0..n as u64 {
        let x = insert(j, forest.vars, |bit| {
            if bit >= forest.queries && bit < forest.queries + gt {
                Some((token >> (bit - forest.queries)) & 1)
            } else if bit >= forest.layers && bit < forest.layers + gl {
                Some((layer >> (bit - forest.layers)) & 1)
            } else {
                None
            }
        });
        share.push(forest.datum(f, x));
        forest_indices.push(ForestRecord {
            group: h as u64,
            position: j,
            original: x,
        });
    }
    Ok(GroupOutput {
        columns,
        forest: share,
        indices,
        forest_indices,
    })
}
