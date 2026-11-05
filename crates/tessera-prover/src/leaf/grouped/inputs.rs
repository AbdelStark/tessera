// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Owned group inputs with separate resident and scoped split oracles (RFC-0077).

use super::*;
use crate::leaf::GroupRead;

#[derive(Clone, Copy)]
pub(crate) struct ForestShape {
    pub(crate) vars: u32,
    pub(crate) height: u32,
    pub(crate) queries: u32,
    pub(crate) layers: u32,
}

enum Splits<'a, S> {
    Resident {
        stage: &'a LeafLayout<S>,
        witness: &'a [Vec<SiteTensors<'a, S>>],
        forests: &'a [StageForest<'a>],
    },
    Scoped(Vec<SecretBuf<Goldilocks>>),
}

pub(in crate::leaf) struct GroupInputs<'a, S> {
    pub(in crate::leaf) layout: GroupLayout,
    pub(in crate::leaf) columns: GroupColumns,
    pub(in crate::leaf) shares: Vec<SecretBuf<Goldilocks>>,
    group: u64,
    splits: Splits<'a, S>,
}

impl<'a, S: Copy + Eq> GroupInputs<'a, S> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::leaf) fn resident(
        group: usize,
        layout: GroupLayout,
        columns: GroupColumns,
        shares: Vec<SecretBuf<Goldilocks>>,
        stage: &'a LeafLayout<S>,
        witness: &'a [Vec<SiteTensors<'a, S>>],
        forests: &'a [StageForest<'a>],
    ) -> Self {
        Self {
            layout,
            columns,
            shares,
            group: group as u64,
            splits: Splits::Resident {
                stage,
                witness,
                forests,
            },
        }
    }

    pub(in crate::leaf) fn scoped(
        group: usize,
        layout: GroupLayout,
        columns: GroupColumns,
        shares: Vec<SecretBuf<Goldilocks>>,
        splits: Vec<SecretBuf<Goldilocks>>,
    ) -> Result<Self, LeafError> {
        if splits.len() != layout.reads().len() {
            return internal("one scoped source per group split");
        }
        Ok(Self {
            layout,
            columns,
            shares,
            group: group as u64,
            splits: Splits::Scoped(splits),
        })
    }

    pub(in crate::leaf) fn split(
        &self,
        slot: usize,
        read: &GroupRead,
        free: &[GoldilocksExt2],
        original: &[GoldilocksExt2],
    ) -> Result<GoldilocksExt2, LeafError> {
        if self.layout.reads().get(slot) != Some(read) {
            return internal("a group split keeps its complete source identity");
        }
        match &self.splits {
            Splits::Resident { stage, witness, .. } => {
                let source_read = stage
                    .reads(read.site, read.layer)
                    .and_then(|r| r.get(read.read))
                    .ok_or(LeafError::Internal {
                        rule: "a group's split reads the stage's tensors",
                    })?;
                let values = witness
                    .get(read.site)
                    .and_then(|s| s.get(read.layer))
                    .ok_or(LeafError::Internal {
                        rule: "a group's source has its original site and layer",
                    })?
                    .of(source_read)?;
                let mut table =
                    SecretBuf::try_with_capacity(values.len()).map_err(|_| LeafError::Memory)?;
                table.extend(values.iter().map(|&v| Goldilocks::from_i64(v)));
                evaluate_base::<GoldilocksExt2>(&table, original).map_err(|_| LeafError::Internal {
                    rule: "a tensor's point has its variables",
                })
            }
            Splits::Scoped(tables) => {
                let table = tables.get(slot).ok_or(LeafError::Internal {
                    rule: "a group's split has its scoped table",
                })?;
                evaluate_base::<GoldilocksExt2>(table, free).map_err(|_| LeafError::Internal {
                    rule: "a scoped tensor's point has its free variables",
                })
            }
        }
    }

    pub(in crate::leaf) fn block(
        &self,
        polynomial: usize,
        tensor_count: usize,
        shape: &MergePolynomial,
        bits: u64,
    ) -> Result<Secret<Vec<Goldilocks>>, LeafError> {
        let read = self
            .layout
            .reads()
            .iter()
            .enumerate()
            .find(|(_, r)| r.polynomial == polynomial);
        match &self.splits {
            Splits::Resident {
                stage,
                witness,
                forests,
            } => {
                let source = match read {
                    Some((_, r)) => {
                        let read = stage
                            .reads(r.site, r.layer)
                            .and_then(|reads| reads.get(r.read))
                            .ok_or(LeafError::Internal {
                                rule: "a group's block names its original read",
                            })?;
                        Lent::Tensor(witness[r.site][r.layer].of(read)?)
                    }
                    None => Lent::Forest(
                        forests
                            .get(
                                polynomial
                                    .checked_sub(tensor_count)
                                    .ok_or(LeafError::Memory)?,
                            )
                            .ok_or(LeafError::Internal {
                                rule: "a group's block names its original forest",
                            })?
                            .leaves,
                    ),
                };
                let entry = |x| match source {
                    Lent::Tensor(v) => v.get(x).map(|&v| Goldilocks::from_i64(v)),
                    Lent::Forest(v) => v.get(x).copied(),
                };
                block(entry, shape, bits)
                    .map(Secret::new)
                    .ok_or(LeafError::Internal {
                        rule: "a polynomial's table holds its blocks",
                    })
            }
            Splits::Scoped(tables) => {
                let values = match read {
                    Some((slot, read)) if read.group_bits == bits => tables.get(slot),
                    Some(_) => None,
                    None if bits == self.group => self.shares.get(
                        polynomial
                            .checked_sub(tensor_count)
                            .ok_or(LeafError::Memory)?,
                    ),
                    None => None,
                }
                .ok_or(LeafError::Internal {
                    rule: "a scoped merge block has its polynomial and fixed coordinates",
                })?;
                let free = shape
                    .vars
                    .checked_sub(shape.group_vars.len())
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or(LeafError::Memory)?;
                if Some(values.len()) != 1usize.checked_shl(free) {
                    return internal("a scoped merge block holds its free variables");
                }
                seal_copy(values)
            }
        }
    }
}

pub(in crate::leaf) fn seal_copy(
    values: &[Goldilocks],
) -> Result<Secret<Vec<Goldilocks>>, LeafError> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(values.len())
        .map_err(|_| LeafError::Memory)?;
    // Reserve before the Copy-only fill; no callback/fallible work before its clearing owner.
    copy.extend_from_slice(values);
    Ok(Secret::new(copy))
}
