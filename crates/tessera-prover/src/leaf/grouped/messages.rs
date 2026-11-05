// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0018/0054 grouped message ownership, separate from proving working sets.

use core::mem::size_of;
use core::ops::Range;

use tessera_field::GoldilocksExt2 as F;
use tessera_logup::{LookupClaims, ProvenGroup};
use tessera_protocol::{GroupProof, TreeProof};
use tessera_sumcheck::{Claim, LevelClaim, MergeClaim, MergePolynomial, ProvenTreeLevel};

use super::{LeafError, ProvenGroupedLeaf};

/// Public upper bounds on the counts of one grouped leaf stage's messages.
///
/// Counts cover every group, including an exceptional token-zero subset. `max_vars`
/// bounds every tree height, forest leaf dimension and merged polynomial dimension.
/// Derive these bounds from the public layout/schedule before P1. This object does
/// not authenticate a caller's correspondence with a layout or witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedMessageShape {
    /// Total number of groups, at least two for RFC-0054 grouped messages.
    pub groups: u64,
    /// Maximum lookup instances in any group and in the table batch.
    pub lookups: u64,
    /// Maximum product forests in any group, also bounding all forest polynomials.
    pub forests: u64,
    /// Maximum tensor split values in any group, also bounding tensor polynomials.
    pub tensors: u64,
    /// Maximum variables of any tree/forest/merged polynomial before grouping.
    pub max_vars: u32,
    /// Number of graph atom split values retained by the CPU driver.
    pub atom_splits: u64,
}

/// Requested heap bounds for grouped messages and the CPU's wire copies.
///
/// Includes nested vectors, spare capacities, original claim points, merge input
/// metadata and the separately retained CPU atom splits. Borrowed layout/witness
/// storage, tree/table/merge payloads and kernel scratch, transcript callbacks,
/// opening/stage work, stack and allocator/process overhead remain additional.
/// This component alone admits no physical CPU route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedMessageMemory {
    retained: u64,
    wire: u64,
    driver: u64,
}

impl GroupedMessageMemory {
    /// Price message owners without allocation, witnesses or transcript challenges.
    ///
    /// # Errors
    /// [`LeafError::Memory`] for overflow or an unaddressable request;
    /// [`LeafError::Internal`] for a non-grouped count or an unsupported dimension.
    pub fn new(shape: GroupedMessageShape) -> Result<Self, LeafError> {
        if shape.groups < 2 || !shape.groups.is_power_of_two() || shape.groups > 1 << 14 {
            return super::super::internal("grouped messages require 1..14 group bits");
        }
        if shape.max_vars as usize > tessera_sumcheck::MAX_MERGE_VARS {
            return super::super::internal("grouped message dimensions fit the merge");
        }
        let GroupedMessageShape {
            groups,
            lookups,
            forests,
            tensors,
            max_vars,
            atom_splits,
        } = shape;
        let vars = u64::from(max_vars);
        let coords = add(vars, 1)?; // include a next-level claim's pair coordinate
        let trees = add(lookups, forests)?;
        let polys = add(tensors, forests)?;
        let group = sum([
            levels(vars, trees)?,
            growing::<F>(mul(2, lookups)?)?,
            growing::<F>(lookups)?,
            growing::<LookupClaims<F>>(lookups)?,
            mul(mul(2, lookups)?, growing::<F>(coords)?)?,
            growing::<Claim<F>>(forests)?,
            mul(forests, growing::<F>(coords)?)?,
            growing::<F>(tensors)?,
            growing::<F>(coords)?,
        ])?;
        let tables = sum([
            levels(vars, lookups)?,
            growing::<F>(mul(2, lookups)?)?,
            growing::<Claim<F>>(lookups)?,
            mul(lookups, growing::<F>(coords)?)?,
        ])?;
        let retained = sum([
            size_of::<ProvenGroupedLeaf>() as u64, // CPU retains it in a Box
            growing::<ProvenGroup<F>>(groups)?,
            mul(2, growing::<Vec<F>>(groups)?)?, // splits and leaf points
            mul(groups, group)?,
            tables,
            rounds(vars, polys)?,
            growing::<F>(vars)?,
            growing::<usize>(polys)?,
            growing::<Range<usize>>(polys)?,
            growing::<Claim<F>>(polys)?,
            mul(polys, growing::<F>(vars)?)?,
            growing::<F>(atom_splits)?,
        ])?;
        let group_wire = sum([
            growing::<F>(mul(2, lookups)?)?,
            growing::<tessera_sumcheck::SumcheckProof<F>>(vars)?,
            mul(vars, rounds(vars, mul(4, trees)?)?)?,
            growing::<F>(lookups)?,
            growing::<F>(tensors)?,
        ])?;
        let wire = sum([
            growing::<GroupProof>(groups)?,
            mul(groups, group_wire)?,
            growing::<tessera_sumcheck::SumcheckProof<F>>(vars)?,
            mul(vars, rounds(vars, mul(4, lookups)?)?)?,
            growing::<F>(mul(2, lookups)?)?,
            rounds(vars, polys)?,
            growing::<F>(atom_splits)?,
        ])?;
        let driver = sum([
            growing::<super::ForestShape>(forests)?,
            growing::<MergePolynomial>(polys)?,
            mul(polys, growing::<usize>(vars)?)?,
            growing::<(usize, usize, usize)>(tensors)?,
            growing::<Vec<MergeClaim<F>>>(groups)?,
            mul(groups, growing::<MergeClaim<F>>(polys)?)?,
            mul(mul(groups, polys)?, growing::<F>(vars)?)?,
            growing::<F>(groups)?,
            mul(2, growing::<F>(coords)?)?, // current free/full split coordinates
        ])?;
        // Check the complete message-owner overlap as well as its components.
        sum([retained, wire, driver])?;
        Ok(Self {
            retained,
            wire,
            driver,
        })
    }

    /// Returned grouped proof, CPU Box and retained atom splits.
    pub fn retained_bytes(self) -> u64 {
        self.retained
    }

    /// Additional storage allocated by [`ProvenGroupedLeaf::to_wire`].
    pub fn wire_bytes(self) -> u64 {
        self.wire
    }

    /// Driver polynomial/read/claim/gamma metadata; payload work is additional.
    pub fn driver_bytes(self) -> u64 {
        self.driver
    }

    /// Conservative coexistence of all priced owners, without payload work.
    pub fn overlap_bytes(self) -> u64 {
        self.retained + self.wire + self.driver
    }
}

impl ProvenGroupedLeaf {
    /// Clone this grouped stage's protocol messages and atom splits for Round2.
    /// Original tree/merge claims and their points remain owned by `self`.
    pub fn to_wire(&self, atoms: &[F]) -> TreeProof {
        TreeProof {
            levels: self
                .tables()
                .levels()
                .iter()
                .map(|l| l.proof().clone())
                .collect(),
            groups: self
                .groups()
                .iter()
                .zip(self.splits())
                .map(|(group, splits)| GroupProof {
                    roots: group.roots().to_vec(),
                    levels: group.levels().iter().map(|l| l.proof().clone()).collect(),
                    values: group.values().to_vec(),
                    splits: splits.clone(),
                })
                .collect(),
            table_roots: self.tables().roots().to_vec(),
            merge: self.merge().proof().clone(),
            splits: atoms.to_vec(),
        }
    }
}

fn add(a: u64, b: u64) -> Result<u64, LeafError> {
    a.checked_add(b).ok_or(LeafError::Memory)
}
fn mul(a: u64, b: u64) -> Result<u64, LeafError> {
    a.checked_mul(b).ok_or(LeafError::Memory)
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64, LeafError> {
    values.into_iter().try_fold(0, add)
}
fn array<T>(n: u64) -> Result<u64, LeafError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(LeafError::Memory);
    }
    Ok(bytes)
}
fn growing<T>(n: u64) -> Result<u64, LeafError> {
    if n == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    // A conservative doubling-growth envelope, including the old allocation.
    array::<T>(mul(4, n.max(minimum))?)
}
fn rounds(vars: u64, evaluations: u64) -> Result<u64, LeafError> {
    sum([
        growing::<Vec<F>>(vars)?,
        mul(vars, growing::<F>(3)?)?,
        growing::<F>(evaluations)?,
    ])
}
fn levels(vars: u64, trees: u64) -> Result<u64, LeafError> {
    if trees == 0 {
        return Ok(0);
    }
    let level = sum([
        mul(2, growing::<F>(add(vars, 1)?)?)?,
        rounds(vars, mul(4, trees)?)?,
        growing::<usize>(trees)?,
        growing::<LevelClaim<F>>(trees)?,
        mul(trees, growing::<F>(add(vars, 1)?)?)?,
    ])?;
    add(growing::<ProvenTreeLevel<F>>(vars)?, mul(vars, level)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_field::Field;
    use tessera_sumcheck::Merge;

    fn shape() -> GroupedMessageShape {
        GroupedMessageShape {
            groups: 2,
            lookups: 0,
            forests: 0,
            tensors: 0,
            max_vars: 0,
            atom_splits: 0,
        }
    }

    #[test]
    fn public_message_sizing_allocates_nothing_and_refuses_invalid_shapes() {
        let measured = allocation_counter::measure(|| {
            for groups in [2, 4, 1 << 14] {
                let memory =
                    GroupedMessageMemory::new(GroupedMessageShape { groups, ..shape() }).unwrap();
                assert!(memory.overlap_bytes() >= memory.retained_bytes() + memory.wire_bytes());
            }
        });
        assert_eq!(measured.count_total, 0);
        for groups in [0, 1, 3, 1 << 15] {
            assert_eq!(
                GroupedMessageMemory::new(GroupedMessageShape { groups, ..shape() }),
                Err(LeafError::Internal {
                    rule: "grouped messages require 1..14 group bits"
                })
            );
        }
        assert_eq!(
            GroupedMessageMemory::new(GroupedMessageShape {
                max_vars: 41,
                ..shape()
            }),
            Err(LeafError::Internal {
                rule: "grouped message dimensions fit the merge"
            })
        );
    }

    #[test]
    fn every_public_count_overflow_is_a_typed_memory_refusal() {
        let cases = [
            GroupedMessageShape {
                lookups: u64::MAX,
                ..shape()
            },
            GroupedMessageShape {
                forests: u64::MAX,
                ..shape()
            },
            GroupedMessageShape {
                tensors: u64::MAX,
                ..shape()
            },
            GroupedMessageShape {
                atom_splits: u64::MAX,
                ..shape()
            },
            GroupedMessageShape {
                groups: 1 << 14,
                tensors: (isize::MAX as u64) / 4096,
                max_vars: 40,
                ..shape()
            },
        ];
        for shape in cases {
            let measured = allocation_counter::measure(|| {
                assert_eq!(GroupedMessageMemory::new(shape), Err(LeafError::Memory))
            });
            assert_eq!(measured.count_total, 0);
        }
    }

    #[test]
    fn public_array_addressability_and_owner_sum_overflow_are_refused() {
        // Each failure is independent: the multiplication here still fits u64.
        assert_eq!(
            array::<F>(isize::MAX as u64 / size_of::<F>() as u64 + 1),
            Err(LeafError::Memory)
        );
        assert_eq!(sum([u64::MAX, 1]), Err(LeafError::Memory));
        assert_eq!(mul(u64::MAX, 2), Err(LeafError::Memory));
    }

    #[test]
    fn retained_merge_input_points_and_tensor_read_metadata_fit_the_driver_bound() {
        for (groups, tensors, vars) in [(2u64, 1, 1), (8, 17, 20), (32, 129, 40)] {
            let bits = groups.trailing_zeros() as usize;
            let shape = GroupedMessageShape {
                groups,
                tensors,
                max_vars: vars,
                ..shape()
            };
            let bound = GroupedMessageMemory::new(shape).unwrap();
            let mut held = None;
            let measured = allocation_counter::measure(|| {
                let polynomials = (0..tensors)
                    .map(|_| MergePolynomial {
                        vars: vars as usize,
                        group_vars: (0..bits).collect(),
                    })
                    .collect();
                let reads: Vec<_> = (0..tensors).map(|i| (i as usize, 0usize, 0usize)).collect();
                let claims = (0..groups)
                    .map(|group| {
                        (0..tensors)
                            .map(|polynomial| MergeClaim {
                                polynomial: polynomial as usize,
                                point: vec![F::ZERO; vars as usize - bits],
                                group_bits: group,
                                value: F::ZERO,
                            })
                            .collect()
                    })
                    .collect();
                let merge =
                    Merge::new(polynomials, claims, vec![F::ONE; groups as usize], F::ONE).unwrap();
                held = Some((merge, reads));
            });
            assert!(
                measured.bytes_max <= bound.driver_bytes(),
                "{measured:?} {bound:?}"
            );
            assert_eq!(held.as_ref().unwrap().0.claims().len(), groups as usize);
        }
    }
}
