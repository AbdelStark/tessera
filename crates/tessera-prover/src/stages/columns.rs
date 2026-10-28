// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public recipes for regenerated stage operands. Dense storage remains the resident oracle.

use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_ir::TensorId;

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
        bit: usize,
        head_vars: usize,
        row_vars: usize,
        rows: usize,
        sine: bool,
        len: usize,
    },
    Function {
        len: usize,
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
        at: impl Fn(usize) -> Result<Goldilocks, StageError> + Send + Sync + 'static,
    ) -> Result<Self, StageError> {
        if streaming {
            Ok(Self::Function {
                len,
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
        Self::generated(streaming, len, move |index| {
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
        })
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
        Self::generated(true, len, move |i| {
            Ok(if i % width >= usize::from(bits) {
                Goldilocks::ONE
            } else {
                Goldilocks::ZERO
            })
        })
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
                bit,
                head_vars,
                row_vars,
                rows,
                sine,
                ..
            } => {
                let row = (index >> (bit + head_vars)) & ((1usize << row_vars) - 1);
                if row >= *rows {
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
