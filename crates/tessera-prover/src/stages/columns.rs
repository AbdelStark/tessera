// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public recipes for regenerated stage operands. Dense storage remains the resident oracle.

use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_ir::TensorId;

use super::source_memory::{add, arc, array};
use super::tables::TensorReader;
use super::{StageError, WitnessSource};

pub(super) enum Column {
    Secret(SecretBuf<Goldilocks>),
    Public(Vec<Goldilocks>),
    Tensor {
        tensors: Vec<TensorId>,
        shape: Vec<u32>,
        block: usize,
        len: usize,
        padded: bool,
    },
    Broadcast {
        input: Box<Column>,
        low: usize,
        len: usize,
    },
    Spread {
        input: Box<Column>,
        block: usize,
        rows: usize,
        len: usize,
    },
    Rope {
        table: std::sync::Arc<tessera_tables::RopeTable>,
        positions: TensorId,
        slots: Option<std::sync::Arc<tessera_ir::BatchSlots>>,
        bit: usize,
        head_vars: usize,
        row_vars: usize,
        rows: usize,
        sine: bool,
        len: usize,
    },
    Function {
        len: usize,
        heap: u64,
        at: Box<dyn Fn(usize) -> Result<Goldilocks, StageError> + Send + Sync>,
    },
}

impl From<SecretBuf<Goldilocks>> for Column {
    fn from(values: SecretBuf<Goldilocks>) -> Self {
        Self::Secret(values)
    }
}
impl From<Vec<Goldilocks>> for Column {
    fn from(values: Vec<Goldilocks>) -> Self {
        Self::Public(values)
    }
}

impl Column {
    pub(super) fn generated(
        streaming: bool,
        len: usize,
        owned: u64,
        at: impl Fn(usize) -> Result<Goldilocks, StageError> + Send + Sync + 'static,
    ) -> Result<Self, StageError> {
        if streaming {
            let heap = add(size_of_val(&at) as u64, owned)?;
            Ok(Self::Function {
                len,
                heap,
                at: Box::new(at),
            })
        } else {
            Ok(Self::Public((0..len).map(at).collect::<Result<_, _>>()?))
        }
    }

    pub(super) fn selector(
        shape: &[u32],
        outside: bool,
        streaming: bool,
    ) -> Result<Self, StageError> {
        if !streaming {
            return Ok(super::public::selector(shape, outside)?.into());
        }
        let shape = shape.to_vec();
        let vars: u32 = shape.iter().map(|&n| super::tables::axis_vars(n)).sum();
        let len = 1usize.checked_shl(vars).ok_or(StageError::Memory)?;
        Self::generated(
            streaming,
            len,
            array::<u32>(shape.capacity())?,
            move |index| {
                let mut rest = index;
                let mut real = !shape.is_empty();
                for &n in shape.iter().rev() {
                    let bits = super::tables::axis_vars(n);
                    real &= (rest & ((1usize << bits) - 1)) < n as usize;
                    rest >>= bits;
                }
                Ok(if real != outside {
                    Goldilocks::ONE
                } else {
                    Goldilocks::ZERO
                })
            },
        )
    }

    pub(super) fn bit_slots(shape: &[u32], bits: u8, streaming: bool) -> Result<Self, StageError> {
        if !streaming {
            return Ok(super::public::bit_slots(shape, bits)?.into());
        }
        if shape.last() != Some(&u32::from(bits)) {
            return Err(StageError::Internal {
                rule: "a bit axis is its output's last",
            });
        }
        let len = 1usize
            << shape
                .iter()
                .map(|&n| super::tables::axis_vars(n))
                .sum::<u32>();
        let width = u32::from(bits).next_power_of_two() as usize;
        Self::generated(true, len, 0, move |i| {
            Ok(if i % width >= usize::from(bits) {
                Goldilocks::ONE
            } else {
                Goldilocks::ZERO
            })
        })
    }

    /// Heap below this descriptor. Shared Arc payloads are charged per reference,
    /// conservatively; canonical RopeTable constructors reserve frequencies exactly.
    pub(super) fn heap_bytes(&self) -> Result<u64, StageError> {
        match self {
            Self::Secret(v) => array::<Goldilocks>(v.capacity()),
            Self::Public(v) => array::<Goldilocks>(v.capacity()),
            Self::Tensor { tensors, shape, .. } => add(
                array::<TensorId>(tensors.capacity())?,
                array::<u32>(shape.capacity())?,
            ),
            Self::Broadcast { input, .. } | Self::Spread { input, .. } => {
                add(size_of::<Column>() as u64, input.heap_bytes()?)
            }
            Self::Rope { table, slots, .. } => {
                let rope = add(
                    arc::<tessera_tables::RopeTable>()?,
                    array::<u128>(table.constants().frequencies().len())?,
                )?;
                let slots = slots.as_ref().map_or(Ok(0), |s| {
                    add(
                        arc::<tessera_ir::BatchSlots>()?,
                        array::<u32>(s.lengths.capacity())?,
                    )
                })?;
                add(rope, slots)
            }
            Self::Function { heap, .. } => Ok(*heap),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Secret(values) => values.len(),
            Self::Public(values) => values.len(),
            Self::Tensor { len, .. }
            | Self::Broadcast { len, .. }
            | Self::Spread { len, .. }
            | Self::Function { len, .. }
            | Self::Rope { len, .. } => *len,
        }
    }

    pub(super) fn resident(&self) -> Result<&[Goldilocks], StageError> {
        match self {
            Self::Secret(values) => Ok(values),
            Self::Public(values) => Ok(values),
            _ => Err(StageError::Internal {
                rule: "the resident builder reads materialized columns",
            }),
        }
    }

    pub(super) fn value<W: WitnessSource + ?Sized>(
        &self,
        witness: &W,
        index: usize,
    ) -> Result<Goldilocks, StageError> {
        if index >= self.len() {
            return Err(StageError::Internal {
                rule: "a stage column index fits its public geometry",
            });
        }
        match self {
            Self::Secret(values) => Ok(values[index]),
            Self::Public(values) => Ok(values[index]),
            Self::Tensor {
                tensors,
                shape,
                block,
                padded,
                ..
            } => {
                let layer = index / block;
                match tensors.get(layer) {
                    Some(&tensor) => {
                        Ok(TensorReader::with_padding(witness, tensor, shape, *padded)?
                            .entry(index % block))
                    }
                    None => Ok(Goldilocks::ZERO),
                }
            }
            Self::Broadcast { input, low, .. } => {
                input.value(witness, (index >> low) & (input.len() - 1))
            }
            Self::Spread {
                input, block, rows, ..
            } => input.value(witness, (index / (block * rows)) * block + index % block),
            Self::Rope {
                table,
                positions,
                slots,
                bit,
                head_vars,
                row_vars,
                rows,
                sine,
                ..
            } => {
                let row = (index >> (bit + head_vars)) & ((1usize << row_vars) - 1);
                if row >= *rows || slots.as_ref().is_some_and(|s| !s.is_real_row(row)) {
                    return Ok(Goldilocks::ZERO);
                }
                let values = witness.values(*positions).ok_or(StageError::Internal {
                    rule: "the regenerated witness holds public positions",
                })?;
                let position = values.get(row).and_then(|&v| u32::try_from(v).ok()).ok_or(
                    StageError::Internal {
                        rule: "a public position is a 32-bit identifier",
                    },
                )?;
                let (c, s) = table
                    .constants()
                    .at(position, (index & ((1usize << bit) - 1)) as u32)
                    .ok_or(StageError::Internal {
                        rule: "a Rotate's positions and frequencies fit its table",
                    })?;
                Ok(Goldilocks::from_i64(if *sine { s } else { c }))
            }
            Self::Function { at, .. } => at(index),
        }
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;
    #[test]
    fn source_heap_counts_owned_capture_and_nested_capacity() {
        for count in [1usize, 4, 1024] {
            let mut held = None;
            let measured = allocation_counter::measure(|| {
                let mut values = Vec::with_capacity(count * 2);
                values.resize(count, Goldilocks::ONE);
                let owned = array::<Goldilocks>(values.capacity()).unwrap();
                let function =
                    Column::generated(true, count, owned, move |i| Ok(values[i])).unwrap();
                held = Some(Column::Spread {
                    input: Box::new(Column::Broadcast {
                        input: Box::new(function),
                        low: 0,
                        len: count,
                    }),
                    block: count,
                    rows: 1,
                    len: count,
                });
            });
            assert!(
                u64::try_from(measured.bytes_current).unwrap()
                    <= held.as_ref().unwrap().heap_bytes().unwrap()
            );
        }
        let mut held = None;
        let measured = allocation_counter::measure(|| {
            let mut tensors = Vec::with_capacity(512);
            tensors.push(TensorId(0));
            let mut shape = Vec::with_capacity(256);
            shape.push(1);
            held = Some(Column::Tensor {
                tensors,
                shape,
                block: 1,
                len: 1,
                padded: true,
            });
        });
        assert!(
            u64::try_from(measured.bytes_current).unwrap()
                <= held.as_ref().unwrap().heap_bytes().unwrap()
        );
    }
    #[test]
    fn source_heap_counts_selectors_and_clearing_spare_capacity() {
        let shape = vec![1; 129];
        let mut held = None;
        let measured = allocation_counter::measure(|| {
            held = Some(Column::selector(&shape, true, true).unwrap());
        });
        assert!(
            u64::try_from(measured.bytes_current).unwrap()
                <= held.as_ref().unwrap().heap_bytes().unwrap()
        );
        drop(held.take());
        let measured = allocation_counter::measure(|| {
            let mut v = SecretBuf::filled(Goldilocks::ZERO, 4096);
            v.truncate(1);
            held = Some(Column::Secret(v));
        });
        assert!(
            u64::try_from(measured.bytes_current).unwrap()
                <= held.as_ref().unwrap().heap_bytes().unwrap()
        );
    }
    #[test]
    fn source_heap_counts_rope_slots_and_public_spare_capacity() {
        let mut held = None;
        let measured = allocation_counter::measure(|| {
            let table = tessera_tables::RopeTable::new(
                tessera_tables::TableId(0),
                tessera_tables::TableKind::Rope {
                    theta: 10000,
                    factor: 1,
                    low_freq_factor: 1,
                    high_freq_factor: 4,
                    original_max: 8,
                    bits: 14,
                    max_context: 2,
                    d_head: 256,
                },
            )
            .unwrap();
            let slots = tessera_ir::BatchSlots::new(vec![1; 4097]).unwrap();
            held = Some(Column::Rope {
                table: std::sync::Arc::new(table),
                slots: Some(std::sync::Arc::new(slots)),
                positions: TensorId(0),
                bit: 7,
                head_vars: 0,
                row_vars: 13,
                rows: 4097,
                sine: false,
                len: 1 << 20,
            });
        });
        assert!(
            u64::try_from(measured.bytes_current).unwrap()
                <= held.as_ref().unwrap().heap_bytes().unwrap()
        );
        drop(held.take());
        let measured = allocation_counter::measure(|| {
            let mut values = Vec::with_capacity(4096);
            values.push(Goldilocks::ZERO);
            held = Some(Column::Public(values));
        });
        assert!(
            u64::try_from(measured.bytes_current).unwrap()
                <= held.as_ref().unwrap().heap_bytes().unwrap()
        );
    }
}
