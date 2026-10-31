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
impl StageSchedule {
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
