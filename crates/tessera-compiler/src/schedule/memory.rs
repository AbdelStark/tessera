// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained public schedule metadata, including nested spare capacities.
use super::*;

fn vector<T>(v: &Vec<T>) -> Option<u64> {
    (v.capacity() as u64).checked_mul(size_of::<T>() as u64)
}
fn layer(value: &Layers) -> Option<u64> {
    match value {
        Layers::Eq(point) => vector(point),
        Layers::None | Layers::Each | Layers::One(_) => Some(0),
    }
}
fn coefficient(value: &Vec<Vec<Factor>>) -> Option<u64> {
    let mut bytes = vector(value)?;
    for product in value {
        bytes = bytes.checked_add(vector(product)?)?;
        for factor in product {
            bytes = bytes.checked_add(match factor {
                Factor::Lt { coords, len: _ } => vector(coords)?,
                Factor::Minus | Factor::Pow2(_) | Factor::Constant(_) => 0,
            })?;
        }
    }
    Some(bytes)
}
fn atom(value: &Atom) -> Option<u64> {
    let Atom {
        kind: _,
        point,
        layers,
        coefficient: terms,
        class: _,
        combined,
    } = value;
    let mut bytes = vector(point)?
        .checked_add(layer(layers)?)?
        .checked_add(coefficient(terms)?)?
        .checked_add(vector(combined)?)?;
    for Member {
        kind: _,
        coefficient: terms,
    } in combined
    {
        bytes = bytes.checked_add(coefficient(terms)?)?;
    }
    Some(bytes)
}
fn claim(value: &Claim) -> Option<u64> {
    let Claim {
        kind: _,
        point,
        layers,
        atoms,
        public,
        split: _,
        parts,
    } = value;
    let mut bytes = vector(point)?
        .checked_add(layer(layers)?)?
        .checked_add(vector(atoms)?)?
        .checked_add(vector(public)?)?
        .checked_add(vector(parts)?)?;
    for value in atoms {
        bytes = bytes.checked_add(atom(value)?)?;
    }
    for PublicTerm {
        source: _,
        point,
        coefficient: terms,
    } in public
    {
        bytes = bytes
            .checked_add(vector(point)?)?
            .checked_add(coefficient(terms)?)?;
    }
    for part in parts {
        bytes = bytes.checked_add(match part {
            Part::Pool { atoms, claim: _ } => vector(atoms)?,
            Part::Instance {
                atom: _,
                instance: _,
            } => 0,
        })?;
    }
    Some(bytes)
}
fn product_heap(product: &Vec<Factor>) -> Option<u64> {
    let mut bytes = vector(product)?;
    for factor in product {
        if let Factor::Lt { coords, .. } = factor {
            bytes = bytes.checked_add(vector(coords)?)?;
        }
    }
    Some(bytes)
}

impl StageSchedule {
    /// Conservative heap peak of canonical schedule derivation over the same
    /// validated graph (RFC-0018/0037/0072). Counts output/growth, public indexes,
    /// relation planning, raw atoms before combination and recursive substitution
    /// frames. Borrowed graph/table/native descriptors and allocator overhead are
    /// separate. Does not allocate or authenticate the pairing. `None` means overflow.
    pub fn construction_peak_bytes(&self, graph: &Graph) -> Option<u64> {
        use crate::memory::{array, grow, map, sum};
        let nt = graph.tensors.len() as u64;
        let nn = graph.nodes.len() as u64;
        let rank = graph
            .tensors
            .iter()
            .map(|t| t.shape.len() as u64)
            .max()
            .unwrap_or(0);
        let mut coords = u64::from(self.layers.vars);
        for tensor in &graph.tensors {
            let own = tensor
                .shape
                .iter()
                .try_fold(0u64, |n, &d| n.checked_add(u64::from(axis_vars(d))))?;
            coords = coords.max(own.checked_add(u64::from(self.layers.vars))?);
        }
        let mut raw = 0u64;
        let mut product = 0u64;
        let mut claim_peak = 0u64;
        for stage in &self.stages {
            for value in &stage.claims {
                claim_peak = claim_peak.max(claim(value)?);
                let mut own_raw = 0u64;
                for atom in &value.atoms {
                    // merge() coalesces equal atoms immediately. combine() only
                    // shares the members' points, preserving their identities.
                    own_raw = own_raw.checked_add((atom.combined.len() as u64).max(1))?;
                    for coefficient in core::iter::once(&atom.coefficient)
                        .chain(atom.combined.iter().map(|m| &m.coefficient))
                    {
                        for entry in coefficient {
                            product = product.max(product_heap(entry)?);
                        }
                    }
                }
                for term in &value.public {
                    own_raw = own_raw.checked_add(1)?;
                    for entry in &term.coefficient {
                        product = product.max(product_heap(entry)?);
                    }
                }
                // One trial atom may coexist with its already merged predecessor.
                raw = raw.max(own_raw.checked_add(1)?);
            }
        }
        // Merge appends every coefficient product; none cancels out. Each prefix
        // fits a final product, including Lt's coordinates. Vec cloning may trim
        // spare capacity, so allow original geometric growth in each frame.
        let frame = sum([
            product.checked_mul(4)?,
            grow::<Coord>(coords)?.checked_mul(2)?,
            grow::<usize>(rank)?,
            grow::<u32>(rank)?,
        ])?;
        sum([
            graph.validation_memory_bytes()?,
            self.heap_bytes()?.checked_mul(2)?,
            self.layers.construction_peak_bytes(graph)?,
            array::<Option<(NodeId, u32)>>(nt)?.checked_mul(3)?,
            array::<bool>(nt)?.checked_mul(6)?,
            map::<tessera_ir::TableId, bool>(nn)?.checked_mul(2)?,
            // padding::zero_value generates only arithmetic tables here;
            // Requant's validated kind clone owns at most four class bytes.
            array::<u8>(4)?,
            array::<(TensorId, usize)>(2)?,
            grow::<Site>(
                graph
                    .nodes
                    .iter()
                    .try_fold(0u64, |n, node| n.checked_add(node.inputs.len() as u64))?
                    .checked_add(1)?,
            )?,
            grow::<usize>((self.stages.len() as u64).checked_add(1)?)?,
            relations::construction_scratch_bytes(self, graph, coords)?,
            claims::construction_scratch_bytes(raw, coords, u64::from(self.layers.vars))?,
            // Original/combined/pool claim payloads can coexist during cloning.
            claim_peak.checked_mul(3)?,
            nn.checked_add(1)?.checked_mul(frame)?,
        ])
    }

    /// Requested heap bytes owned by the schedule, including every nested vector's
    /// spare capacity and its layer map (RFC-0037/0018). Reads public metadata only;
    /// does not allocate, validate, or price construction scratch, the inline object,
    /// runtime/allocator overhead, or any driver state or tensor payload.
    /// Returns `None` if a checked byte count or sum overflows.
    pub fn heap_bytes(&self) -> Option<u64> {
        let Self {
            layers,
            sites,
            stages,
            pool,
        } = self;
        let mut bytes = layers
            .heap_bytes()?
            .checked_add(vector(sites)?)?
            .checked_add(vector(stages)?)?
            .checked_add(vector(pool)?)?;
        for Stage {
            index: _,
            instances,
            rounds: _,
            zero_vars: _,
            claims,
        } in stages
        {
            bytes = bytes
                .checked_add(vector(instances)?)?
                .checked_add(vector(claims)?)?;
            for Instance {
                source: _,
                relation: _,
                node: _,
                kind: _,
                point,
                layers,
                vars: _,
                challenges,
                claims: _,
            } in instances
            {
                bytes = bytes
                    .checked_add(vector(point)?)?
                    .checked_add(layer(layers)?)?
                    .checked_add(vector(challenges)?)?;
            }
            for value in claims {
                bytes = bytes.checked_add(claim(value)?)?;
            }
        }
        for PoolClaim {
            commitment: _,
            point,
            atoms,
            stage: _,
            claim: _,
        } in pool
        {
            bytes = bytes
                .checked_add(vector(point)?)?
                .checked_add(vector(atoms)?)?;
            for value in atoms {
                bytes = bytes.checked_add(atom(value)?)?;
            }
        }
        Some(bytes)
    }
}
