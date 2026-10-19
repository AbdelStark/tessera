// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0050's complete model-tile columns, using the scalar gadgets' honest assignments.

use super::*;
use tessera_compiler::CompareColumn as Array;
use tessera_gadgets::{
    Bracket, CompareColumn, RsqrtColumn, RsqrtRow, RsqrtTerm, Side, Slack, SoftmaxColumn,
    SoftmaxLookup, SoftmaxRow,
};

const SCHEMA: &str = "a selected gadget has its complete canonical column and lookup schema";
const SAME: &str = "a scalar gadget's outputs equal the integer engine's outputs";

pub(super) enum Prepared {
    Root(Box<Root>),
    Softmax(Box<Softmax>),
}

pub(super) struct Root {
    gadget: RsqrtRow,
    epsilon: u64,
    lookups: Vec<tessera_gadgets::Lookup<Goldilocks, RsqrtTerm>>,
    rows: Vec<(TableId, u64)>,
}

pub(super) struct Softmax {
    gadget: SoftmaxRow,
    committed: Vec<SoftmaxColumn>,
    lookups: Vec<SoftmaxLookup<Goldilocks>>,
    rows: Vec<(TableId, u64)>,
}

fn internal<T>() -> Result<T, WitnessError> {
    Err(WitnessError::Internal { rule: SCHEMA })
}

pub(super) fn requant_role(column: Committed) -> Result<AuxColumn, WitnessError> {
    match column {
        Committed::ZLo => Ok(AuxColumn::Low),
        Committed::H0 => Ok(AuxColumn::Middle),
        Committed::G => Ok(AuxColumn::Saturation),
        _ => internal(),
    }
}

fn root_role(column: RsqrtColumn) -> Result<AuxColumn, WitnessError> {
    Ok(match column {
        RsqrtColumn::InputLimb(j) => AuxColumn::InputLimb(u32::from(j)),
        RsqrtColumn::RootLimb(j) => AuxColumn::RootByte(u32::from(j)),
        RsqrtColumn::Compare(bracket, column) => {
            let (column, limb) = match column {
                CompareColumn::Digit(Side::Left, j) => (Array::LeftByte, j),
                CompareColumn::Carry(Side::Left, j) => (Array::LeftCarry, j),
                CompareColumn::Digit(Side::Right, j) => (Array::RightByte, j),
                CompareColumn::Carry(Side::Right, j) => (Array::RightCarry, j),
                CompareColumn::Difference(j) => (Array::Difference, j),
                CompareColumn::Borrow(j) => (Array::Borrow, j),
                _ => return internal(),
            };
            AuxColumn::Comparison {
                upper: bracket == Bracket::Upper,
                column,
                limb: u32::try_from(limb).map_err(|_| WitnessError::Internal { rule: SCHEMA })?,
            }
        }
        _ => return internal(),
    })
}

fn softmax_role(column: SoftmaxColumn) -> Result<AuxColumn, WitnessError> {
    match column {
        SoftmaxColumn::Maximum => Ok(AuxColumn::Maximum),
        SoftmaxColumn::Limb(slack, limb) => Ok(AuxColumn::Slack {
            second: slack == Slack::B,
            limb: u32::from(limb),
        }),
        _ => internal(),
    }
}

impl Prepared {
    pub(super) fn new(
        node: &Node,
        selection: &Selection,
        max_context: u32,
    ) -> Result<Option<Self>, WitnessError> {
        let selected = selection.of(node.id).map(|(_, gadget)| gadget);
        let result = match (&node.op, selected) {
            (
                Op::RsqrtRow {
                    dim,
                    kappa,
                    eps_int,
                },
                Some(Gadget::RsqrtRow),
            ) => {
                let gadget = RsqrtRow::new(*dim, *kappa, *eps_int, &selection.tables)?;
                let rows = gadget
                    .tables()
                    .iter()
                    .map(|t| (t.spec().id, t.row_count()))
                    .collect();
                let constraints = gadget.constraints::<Goldilocks>()?;
                Some(Self::Root(Box::new(Root {
                    gadget,
                    epsilon: *eps_int,
                    lookups: constraints.lookups,
                    rows,
                })))
            }
            (Op::SoftmaxRow { sigma, exp_table }, Some(Gadget::SoftmaxRow { exp, limb, .. }))
                if exp == *exp_table =>
            {
                let width = match limb {
                    Some(id) => match selection
                        .tables
                        .iter()
                        .find(|t| t.id == id)
                        .map(|t| &t.kind)
                    {
                        Some(TableKind::Range { bits }) => *bits,
                        _ => return internal(),
                    },
                    None => sigma
                        .checked_add(18)
                        .ok_or(WitnessError::Internal { rule: SCHEMA })?,
                };
                let gadget = SoftmaxRow::new(exp, width, max_context as usize, &selection.tables)?;
                let rows = gadget
                    .tables()
                    .iter()
                    .map(|t| (t.spec().id, t.row_count()))
                    .collect();
                let constraints = gadget.constraints::<Goldilocks>();
                Some(Self::Softmax(Box::new(Softmax {
                    gadget,
                    committed: constraints.committed,
                    lookups: constraints.lookups,
                    rows,
                })))
            }
            (Op::RsqrtRow { .. } | Op::SoftmaxRow { .. }, _) => return internal(),
            _ => None,
        };
        if let Some(prepared) = &result {
            let actual: Vec<_> = match prepared {
                Self::Root(data) => data.lookups.iter().map(|l| l.table).collect(),
                Self::Softmax(data) => data.lookups.iter().map(|l| l.table).collect(),
            };
            if selection.reads(node.id) != Some(actual.as_slice()) {
                return internal();
            }
        }
        Ok(result)
    }

    pub(super) fn rows(&self) -> &[(TableId, u64)] {
        match self {
            Self::Root(data) => &data.rows,
            Self::Softmax(data) => &data.rows,
        }
    }

    pub(super) fn root_tile(
        &self,
        tensor: TensorId,
        sums: &[i64],
        output: usize,
        tile: &mut TileWitness,
    ) -> Result<(), WitnessError> {
        let Self::Root(data) = self else {
            return internal();
        };
        let mut input = SecretBuf::try_with_capacity(sums.len())
            .map_err(|_| WitnessError::Memory { tensor })?;
        let mut fits = true;
        for &sum in sums {
            let u = i128::from(sum) + i128::from(data.epsilon);
            let valid = (u >= i128::from(data.epsilon)) & (u < 1 << 55);
            fits &= valid;
            input.push(core::hint::select_unpredictable(valid, u as i64, 0));
        }
        if !fits {
            return internal();
        }
        let real = vec![true; sums.len()];
        let witness = data.gadget.witness(&input, &real)?;
        let root = witness
            .tensor(RsqrtColumn::Root)
            .ok_or(WitnessError::Internal { rule: SCHEMA })?;
        if !same(root, &tile.shares[output]) {
            return Err(WitnessError::Internal { rule: SAME });
        }
        for lookup in &data.lookups {
            let rows = data
                .rows
                .iter()
                .find(|(id, _)| *id == lookup.table)
                .map(|(_, rows)| *rows)
                .ok_or(WitnessError::Internal { rule: SCHEMA })?;
            let columns = lookup
                .key
                .terms
                .iter()
                .map(|&(_, term)| {
                    let values = match term {
                        RsqrtTerm::Input => Some(&input[..]),
                        RsqrtTerm::RowSelector => None,
                        RsqrtTerm::Column(c) => Some(
                            witness
                                .tensor(c)
                                .ok_or(WitnessError::Internal { rule: SCHEMA })?,
                        ),
                    };
                    Ok((term, values))
                })
                .collect::<Result<Vec<_>, WitnessError>>()?;
            let keys = keys_of(
                tensor,
                &lookup.key,
                input.len(),
                rows,
                &|term, i| match term {
                    RsqrtTerm::RowSelector => Goldilocks::ONE,
                    _ => Goldilocks::from_i64(
                        columns
                            .iter()
                            .find(|(t, _)| *t == term)
                            .and_then(|(_, values)| *values)
                            .map_or(0, |values| values[i]),
                    ),
                },
            )?;
            if lookup.value.is_some() {
                return internal();
            }
            tile.lookups.push(Lookup {
                table: lookup.table,
                keys,
                values: None,
            });
        }
        for (column, values) in witness.into_tensors() {
            if column != RsqrtColumn::Root {
                tile.gadget.push((root_role(column)?, values));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn softmax_tile(
        &self,
        tensor: TensorId,
        scores: &[i64],
        n: usize,
        valid: impl Fn(usize) -> usize,
        e: usize,
        r: usize,
        tile: &mut TileWitness,
    ) -> Result<(), WitnessError> {
        let Self::Softmax(data) = self else {
            return internal();
        };
        let count = tile.shares[r].len();
        if n == 0 || scores.len() != count * n || tile.shares[e].len() != scores.len() {
            return internal();
        }
        let memory = |_| WitnessError::Memory { tensor };
        for &column in &data.committed {
            if !matches!(
                column,
                SoftmaxColumn::Exponentials | SoftmaxColumn::Reciprocal
            ) {
                tile.gadget.push((
                    softmax_role(column)?,
                    SecretBuf::try_with_capacity(count).map_err(memory)?,
                ));
            }
        }
        for lookup in &data.lookups {
            if lookup.value.is_some() != lookup.per_position {
                return internal();
            }
            tile.lookups.push(Lookup {
                table: lookup.table,
                keys: SecretBuf::try_with_capacity(if lookup.per_position {
                    scores.len()
                } else {
                    count
                })
                .map_err(memory)?,
                values: lookup.per_position.then_some(Column::Share(e)),
            });
        }
        let mut padded = SecretBuf::try_with_capacity(n.next_power_of_two()).map_err(memory)?;
        padded.extend(core::iter::repeat_n(0, n.next_power_of_two()));
        let mut fits = true;
        for (row, scores) in scores.chunks(n).enumerate() {
            padded[..n].copy_from_slice(scores);
            let witness = data.gadget.witness(&padded, valid(row))?;
            let evaluation = data
                .gadget
                .evaluate::<Goldilocks>(&padded, valid(row), &witness)?;
            let exponentials = witness
                .tensor(SoftmaxColumn::Exponentials)
                .ok_or(WitnessError::Internal { rule: SCHEMA })?;
            let reciprocal = witness
                .tensor(SoftmaxColumn::Reciprocal)
                .ok_or(WitnessError::Internal { rule: SCHEMA })?;
            fits &= same(&exponentials[..n], &tile.shares[e][row * n..(row + 1) * n]);
            fits &= same(reciprocal, &tile.shares[r][row..row + 1]);
            for (index, lookup) in data.lookups.iter().enumerate() {
                let group = evaluation
                    .lookups
                    .groups()
                    .iter()
                    .find(|g| g.table() == lookup.table)
                    .ok_or(WitnessError::Internal { rule: SCHEMA })?;
                let position = group
                    .lookups()
                    .iter()
                    .position(|&i| i == index)
                    .ok_or(WitnessError::Internal { rule: SCHEMA })?;
                let offset: usize = group.lengths()[..position].iter().sum();
                let len = if lookup.per_position { n } else { 1 };
                let rows = data
                    .rows
                    .iter()
                    .find(|(id, _)| *id == lookup.table)
                    .map(|(_, rows)| *rows)
                    .ok_or(WitnessError::Internal { rule: SCHEMA })?;
                for key in &group.keys()[offset..offset + len] {
                    let key = key.to_canonical_u64();
                    fits &= key < rows && key <= u64::from(u32::MAX);
                    tile.lookups[index].keys.push(key as u32);
                }
            }
            for (column, values) in witness.into_tensors() {
                if !matches!(
                    column,
                    SoftmaxColumn::Exponentials | SoftmaxColumn::Reciprocal
                ) {
                    let role = softmax_role(column)?;
                    let target = tile
                        .gadget
                        .iter_mut()
                        .find(|(held, _)| *held == role)
                        .ok_or(WitnessError::Internal { rule: SCHEMA })?;
                    target.1.push(values[0]);
                }
            }
        }
        if !fits {
            return Err(WitnessError::Internal { rule: SAME });
        }
        Ok(())
    }
}

fn same(a: &[i64], b: &[i64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(true, |same, (a, b)| same & (a == b))
}

impl Gadgets {
    pub(super) fn prepared(&self, node: u32) -> Result<&Prepared, WitnessError> {
        self.auxiliary
            .get(node as usize)
            .and_then(Option::as_ref)
            .ok_or(WitnessError::Internal { rule: SCHEMA })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_tables::{GeneratedTable, TableSpec};

    fn range_specs() -> Vec<TableSpec> {
        let specs: Vec<TableSpec> = (1..=32)
            .map(|bits| {
                GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                    .unwrap()
                    .spec()
                    .clone()
            })
            .collect();
        specs
    }

    #[test]
    fn root_input_bounds_and_output_identity_are_enforced() {
        let specs = range_specs();
        let gadget = RsqrtRow::new(4, 8, 1, &specs).unwrap();
        let rows = gadget
            .tables()
            .iter()
            .map(|t| (t.spec().id, t.row_count()))
            .collect();
        let constraints = gadget.constraints::<Goldilocks>().unwrap();
        let prepared = Prepared::Root(Box::new(Root {
            gadget,
            epsilon: 1,
            lookups: constraints.lookups,
            rows,
        }));
        let tile = |value| TileWitness {
            shares: vec![SecretBuf::filled(value, 1)],
            ..TileWitness::empty()
        };
        for input in [i64::MIN, -1, (1 << 55) - 1, i64::MAX] {
            assert_eq!(
                prepared.root_tile(TensorId(1), &[input], 0, &mut tile(512)),
                Err(WitnessError::Internal { rule: SCHEMA })
            );
        }
        assert_eq!(
            prepared.root_tile(TensorId(1), &[0], 0, &mut tile(513)),
            Err(WitnessError::Internal { rule: SAME })
        );
        let mut honest = tile(512);
        prepared
            .root_tile(TensorId(1), &[0], 0, &mut honest)
            .unwrap();
        assert_eq!(honest.gadget.len(), 3 + 1 + 12 * 10);
        assert_eq!(honest.lookups.len(), 6 + 1 + 12 * 10);
    }
    #[test]
    fn softmax_shape_and_both_output_identities_are_enforced() {
        let mut specs = range_specs();
        specs.push(
            GeneratedTable::new(
                TableId(100),
                TableKind::Exp {
                    key_bits: 3,
                    sigma: 2,
                    scale: tessera_codec::Rational::new(1, 1).unwrap(),
                },
            )
            .unwrap()
            .spec()
            .clone(),
        );
        let gadget = SoftmaxRow::new(TableId(100), 4, 2, &specs).unwrap();
        let constraints = gadget.constraints::<Goldilocks>();
        let rows = gadget
            .tables()
            .iter()
            .map(|t| (t.spec().id, t.row_count()))
            .collect();
        let prepared = Prepared::Softmax(Box::new(Softmax {
            gadget,
            committed: constraints.committed,
            lookups: constraints.lookups,
            rows,
        }));
        let tile = |e, r| TileWitness {
            shares: vec![SecretBuf::filled(e, 1), SecretBuf::filled(r, 1)],
            ..TileWitness::empty()
        };
        for (e, r) in [(3, 4), (4, 3)] {
            assert_eq!(
                prepared.softmax_tile(TensorId(1), &[0], 1, |_| 1, 0, 1, &mut tile(e, r)),
                Err(WitnessError::Internal { rule: SAME })
            );
        }
        for scores in [&[][..], &[0, 0][..]] {
            assert_eq!(
                prepared.softmax_tile(TensorId(1), scores, 1, |_| 1, 0, 1, &mut tile(4, 4)),
                Err(WitnessError::Internal { rule: SCHEMA })
            );
        }
        assert_eq!(
            prepared.softmax_tile(TensorId(1), &[0], 0, |_| 1, 0, 1, &mut tile(4, 4)),
            Err(WitnessError::Internal { rule: SCHEMA })
        );
        let mut honest = tile(4, 4);
        prepared
            .softmax_tile(TensorId(1), &[0], 1, |_| 1, 0, 1, &mut honest)
            .unwrap();
        assert_eq!(honest.gadget.len(), 9);
        assert_eq!(honest.lookups.len(), 12);
    }
}
