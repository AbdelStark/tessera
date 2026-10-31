// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The tables a stage reads, from the witness: each tensor padded per dimension with zeros, or
//! with the padded values the witness holds ([`WitnessSource::padded`], RFC-0072 'Padded
//! positions'), a stacked kind's layers one after another, its padded layers zero (RFC-0011 rule
//! 1, RFC-0035, RFC-0037 'Layer variables').

use tessera_codec::SecretBuf;
use tessera_compiler::schedule::{Kind, Layers};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_ir::TensorId;
use tessera_poly::{PolyField, eq_table};

use super::{StageError, StageProver, WitnessSource};

impl<W: WitnessSource + ?Sized> StageProver<'_, W> {
    /// The variables of a tensor of `shape`, padded per dimension.
    pub(super) fn vars_of(shape: &[u32]) -> u32 {
        shape.iter().map(|&len| axis_vars(len)).sum()
    }

    /// The shape of `kind`'s tensors.
    pub(super) fn shape(&self, kind: Kind) -> Result<&[u32], StageError> {
        let tensor = self.tensor(kind, 0)?;
        Ok(&self.graph.tensors[tensor.0 as usize].shape)
    }

    /// The tensor of `kind` in `layer`.
    pub(super) fn tensor(&self, kind: Kind, layer: u32) -> Result<TensorId, StageError> {
        self.schedule
            .layers
            .tensor(self.graph, kind, layer)
            .ok_or(StageError::Internal {
                rule: "a kind names a tensor in every layer",
            })
    }

    /// `kind`'s tensor in `layer`, padded per dimension: with zeros, or with the values the
    /// witness holds there ([`WitnessSource::padded`]).
    pub(super) fn slice(
        &self,
        kind: Kind,
        layer: u32,
    ) -> Result<SecretBuf<Goldilocks>, StageError> {
        let tensor = self.tensor(kind, layer)?;
        let shape = &self.graph.tensors[tensor.0 as usize].shape;
        let mut table = zeros(1usize << Self::vars_of(shape))?;
        self.fill(tensor, shape, &mut table)?;
        Ok(table)
    }

    /// Writes `tensor`, of `shape`, into `out`, its padded table: the witness's padded values
    /// when it holds them, else its real values padded per dimension with zeros.
    fn fill(
        &self,
        tensor: TensorId,
        shape: &[u32],
        out: &mut [Goldilocks],
    ) -> Result<(), StageError> {
        if let Some(values) = self.witness.padded(tensor) {
            if values.len() != out.len() {
                return Err(StageError::Internal {
                    rule: "a tensor's padded values fill its padded shape",
                });
            }
            for (slot, &v) in out.iter_mut().zip(values) {
                *slot = Goldilocks::from_i64(v);
            }
            return Ok(());
        }
        if let Some(values) = self.witness.values(tensor) {
            return pad(values, shape, out);
        }
        let reader = TensorReader::with_padding(self.witness, tensor, shape, false)?;
        for (index, value) in out.iter_mut().enumerate() {
            *value = reader.entry(index);
        }
        Ok(())
    }

    /// The table a claim on `kind` with `layers` reads: the kind stacked over every layer for a
    /// claim at a layer point, the one layer's tensor for a claim on one, the tensor for one
    /// outside the layers.
    pub(super) fn operand(
        &self,
        kind: Kind,
        layers: &Layers,
    ) -> Result<SecretBuf<Goldilocks>, StageError> {
        match layers {
            Layers::Eq(_) => self.stacked(kind),
            Layers::One(layer) => self.slice(kind, *layer),
            Layers::None => self.slice(kind, 0),
            Layers::Each => Err(StageError::Internal {
                rule: "an instance's operand is stacked, one layer's or outside the layers",
            }),
        }
    }

    /// `kind` stacked over the `2^λ` layers, the layer index highest; the padded layers are zero.
    pub(super) fn stacked(&self, kind: Kind) -> Result<SecretBuf<Goldilocks>, StageError> {
        let layers = &self.schedule.layers;
        let shape = self.shape(kind)?.to_vec();
        let block = 1usize << Self::vars_of(&shape);
        let mut table = zeros(block << layers.vars)?;
        for layer in 0..layers.count {
            let tensor = self.tensor(kind, layer)?;
            let at = layer as usize * block;
            self.fill(tensor, &shape, &mut table[at..at + block])?;
        }
        Ok(table)
    }
}

/// `⌈log2 len⌉`.
pub(super) fn axis_vars(len: u32) -> u32 {
    len.next_power_of_two().trailing_zeros()
}

/// A table of `len` zeros.
pub(super) fn zeros(len: usize) -> Result<SecretBuf<Goldilocks>, StageError> {
    let mut table = SecretBuf::try_with_capacity(len).map_err(|_| StageError::Memory)?;
    table.resize(len, Goldilocks::ZERO);
    Ok(table)
}

/// Writes `values`, row-major of `shape`, into `out` padded per dimension: the last axis the
/// lowest bits, each axis its own block of `⌈log2 len⌉` bits.
pub(super) fn pad(values: &[i64], shape: &[u32], out: &mut [Goldilocks]) -> Result<(), StageError> {
    let count: usize = shape.iter().map(|&d| d as usize).product();
    if values.len() != count {
        return Err(StageError::Internal {
            rule: "a tensor holds its shape's number of values",
        });
    }
    let Some((&last, outer)) = shape.split_last() else {
        return Ok(());
    };
    let row = last as usize;
    // Each run of the last axis is contiguous: copy it to the padded index of its first entry.
    for (r, run) in values.chunks(row.max(1)).enumerate() {
        let (mut rest, mut index, mut shift) = (r, 0usize, axis_vars(last));
        for &len in outer.iter().rev() {
            index |= (rest % len as usize) << shift;
            rest /= len as usize;
            shift += axis_vars(len);
        }
        for (slot, &v) in out[index..index + run.len()].iter_mut().zip(run) {
            *slot = Goldilocks::from_i64(v);
        }
    }
    Ok(())
}

/// A borrowed integer slice: model weights retain their original INT8 allocation.
pub(super) enum Integers<'a> {
    Wide(&'a [i64]),
    Int8(&'a [i8]),
}
impl<'a> Integers<'a> {
    pub(super) fn real<W: WitnessSource + ?Sized>(
        source: &'a W,
        tensor: TensorId,
    ) -> Result<Self, StageError> {
        if let Some(values) = source.values(tensor) {
            return Ok(Self::Wide(values));
        }
        source
            .int8_values(tensor)
            .map(Self::Int8)
            .ok_or(StageError::Internal {
                rule: "the witness holds every tensor of the graph",
            })
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Wide(values) => values.len(),
            Self::Int8(values) => values.len(),
        }
    }

    pub(super) fn at(&self, index: usize) -> i64 {
        match self {
            Self::Wide(values) => values[index],
            Self::Int8(values) => i64::from(values[index]),
        }
    }
}

/// A borrowed tensor's padded entries, generated by public index arithmetic. No padded copy is
/// made, including when the source supplies nonzero padded witness positions (RFC-0072).
pub(super) struct TensorReader<'a> {
    values: Integers<'a>,
    shape: &'a [u32],
    padded: bool,
    vars: usize,
}

impl<'a> TensorReader<'a> {
    pub(super) fn new<W: WitnessSource + ?Sized>(
        witness: &'a W,
        tensor: TensorId,
        shape: &'a [u32],
    ) -> Result<Self, StageError> {
        Self::with_padding(witness, tensor, shape, true)
    }

    pub(super) fn with_padding<W: WitnessSource + ?Sized>(
        witness: &'a W,
        tensor: TensorId,
        shape: &'a [u32],
        accept_padded: bool,
    ) -> Result<Self, StageError> {
        let vars = shape.iter().try_fold(0usize, |v, &len| {
            v.checked_add(axis_vars(len) as usize)
                .ok_or(StageError::Memory)
        })?;
        let padded_len = 1usize.checked_shl(vars as u32).ok_or(StageError::Memory)?;
        let (values, padded) = match accept_padded.then(|| witness.padded(tensor)).flatten() {
            Some(values) => (Integers::Wide(values), true),
            None => (Integers::real(witness, tensor)?, false),
        };
        let expected = if padded {
            padded_len
        } else {
            shape.iter().try_fold(1usize, |n, &len| {
                n.checked_mul(len as usize).ok_or(StageError::Memory)
            })?
        };
        if values.len() != expected {
            return Err(StageError::Internal {
                rule: "a tensor's values fill its declared shape",
            });
        }
        Ok(Self {
            values,
            shape,
            padded,
            vars,
        })
    }

    pub(super) fn entry(&self, index: usize) -> Goldilocks {
        if self.padded {
            return Goldilocks::from_i64(self.values.at(index));
        }
        // Match `pad`'s empty-shape handling; compiled tensor shapes are nonempty.
        if self.shape.is_empty() {
            return Goldilocks::ZERO;
        }
        let (mut rest, mut flat, mut stride) = (index, 0usize, 1usize);
        for &len in self.shape.iter().rev() {
            let bits = axis_vars(len);
            let coord = rest & ((1usize << bits) - 1);
            if coord >= len as usize {
                return Goldilocks::ZERO;
            }
            flat += coord * stride;
            stride *= len as usize;
            rest >>= bits;
        }
        Goldilocks::from_i64(self.values.at(flat))
    }

    /// Evaluate from public split EQ weights and borrowed entries. Both EQ halves are public;
    /// no allocation proportional to the tensor is retained for the split values or pool.
    pub(super) fn extension(&self, point: &[GoldilocksExt2]) -> Result<GoldilocksExt2, StageError> {
        if point.len() != self.vars {
            return Err(StageError::Internal {
                rule: "a claim's point has its tensor's coordinates",
            });
        }
        let low = point.len() / 2;
        let eq_low = eq_table(&point[..low]).map_err(|_| StageError::Memory)?;
        let eq_high = eq_table(&point[low..]).map_err(|_| StageError::Memory)?;
        let mut sum = GoldilocksExt2::ZERO;
        for (hi, &high_weight) in eq_high.iter().enumerate() {
            let mut block = GoldilocksExt2::ZERO;
            for (lo, &low_weight) in eq_low.iter().enumerate() {
                block += low_weight.mul_by_base(self.entry((hi << low) | lo));
            }
            sum += high_weight * block;
        }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_field::QuadraticExtension;
    use tessera_poly::evaluate_base;

    struct Source {
        real: Vec<i64>,
        padded: Option<Vec<i64>>,
    }
    impl WitnessSource for Source {
        fn values(&self, _: TensorId) -> Option<&[i64]> {
            Some(&self.real)
        }
        fn padded(&self, _: TensorId) -> Option<&[i64]> {
            self.padded.as_deref()
        }
    }

    struct Int8Source {
        real: Vec<i8>,
        wide: Option<Vec<i64>>,
    }
    impl WitnessSource for Int8Source {
        fn values(&self, _: TensorId) -> Option<&[i64]> {
            self.wide.as_deref()
        }
        fn int8_values(&self, _: TensorId) -> Option<&[i8]> {
            Some(&self.real)
        }
    }

    #[test]
    fn borrowed_int8_sign_padding_and_extensions_match_widened_tensors() {
        for shape in [
            vec![1],
            vec![3],
            vec![3, 5],
            vec![2, 3, 7],
            vec![5, 2, 3, 1],
        ] {
            let count: usize = shape.iter().map(|&n| n as usize).product();
            let vars: usize = shape.iter().map(|&n| axis_vars(n) as usize).sum();
            let bytes: Vec<_> = (0..count)
                .map(|i| [i8::MIN, -1, 0, 1, i8::MAX][i % 5])
                .collect();
            let wide = Source {
                real: bytes.iter().map(|&v| i64::from(v)).collect(),
                padded: None,
            };
            let borrowed = Int8Source {
                real: bytes,
                wide: None,
            };
            let expected = TensorReader::new(&wide, TensorId(0), &shape).unwrap();
            let info = allocation_counter::measure(|| {
                let reader = TensorReader::new(&borrowed, TensorId(0), &shape).unwrap();
                for index in 0..1 << vars {
                    assert_eq!(reader.entry(index), expected.entry(index));
                }
            });
            assert_eq!(
                (info.count_total, info.bytes_max),
                (0, 0),
                "no widened copy"
            );
            let reader = TensorReader::new(&borrowed, TensorId(0), &shape).unwrap();
            for seed in 0..=3 {
                let point: Vec<_> = (0..vars)
                    .map(|j| {
                        GoldilocksExt2::new(
                            Goldilocks::from_u64_reduced((seed + j) as u64),
                            Goldilocks::from_u64_reduced(seed as u64),
                        )
                    })
                    .collect();
                assert_eq!(
                    reader.extension(&point).unwrap(),
                    expected.extension(&point).unwrap()
                );
            }
        }
    }

    #[test]
    fn an_int8_source_preserves_wide_precedence_and_exact_shape_refusals() {
        let mut source = Int8Source {
            real: vec![-128, -1, 127],
            wide: Some(vec![7, 8, 9]),
        };
        let reader = TensorReader::new(&source, TensorId(0), &[3]).unwrap();
        assert_eq!(reader.entry(0), Goldilocks::from_i64(7));
        let bad_shape = StageError::Internal {
            rule: "a tensor's values fill its declared shape",
        };
        source.wide = Some(vec![7, 8]);
        assert_eq!(
            TensorReader::new(&source, TensorId(0), &[3]).err(),
            Some(bad_shape)
        );
        source.wide = None;
        source.real.pop();
        assert_eq!(
            TensorReader::new(&source, TensorId(0), &[3]).err(),
            Some(bad_shape)
        );
    }

    #[test]
    fn streamed_tensor_entries_and_extensions_match_the_literal_padded_cube() {
        for shape in [
            vec![1],
            vec![3],
            vec![3, 5],
            vec![2, 3, 7],
            vec![5, 2, 3, 1],
        ] {
            let vars: usize = shape.iter().map(|&n| axis_vars(n) as usize).sum();
            let count: usize = shape.iter().map(|&n| n as usize).product();
            let real: Vec<_> = (0..count)
                .map(|i| [-32768, -1, 0, 1, 32767][i % 5])
                .collect();
            let mut literal = zeros(1 << vars).unwrap();
            pad(&real, &shape, &mut literal).unwrap();
            for padded in [false, true] {
                // A padded source deliberately disagrees with zero padding at every position.
                let padded_values: Vec<_> = (0..1 << vars).map(|i| 3 * i as i64 - 17).collect();
                let source = Source {
                    real: real.clone(),
                    padded: padded.then_some(padded_values.clone()),
                };
                let expected: Vec<_> = if padded {
                    padded_values
                        .into_iter()
                        .map(Goldilocks::from_i64)
                        .collect()
                } else {
                    literal.iter().copied().collect()
                };
                let reader = TensorReader::new(&source, TensorId(0), &shape).unwrap();
                for (index, &value) in expected.iter().enumerate() {
                    assert_eq!(reader.entry(index), value);
                }
                for seed in 0..=3 {
                    let point: Vec<_> = (0..vars)
                        .map(|j| {
                            if seed < 2 {
                                GoldilocksExt2::from(Goldilocks::from_u64_reduced(seed))
                            } else {
                                GoldilocksExt2::new(
                                    Goldilocks::from_u64_reduced((j + seed as usize) as u64),
                                    Goldilocks::from_u64_reduced((2 * j + 1) as u64),
                                )
                            }
                        })
                        .collect();
                    assert_eq!(
                        reader.extension(&point).unwrap(),
                        evaluate_base(&expected, &point).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn split_evaluation_uses_sqrt_public_weights_instead_of_a_tensor_copy() {
        let shape = [257, 257];
        let source = Source {
            real: vec![3; 257 * 257],
            padded: None,
        };
        let reader = TensorReader::new(&source, TensorId(0), &shape).unwrap();
        let point = vec![GoldilocksExt2::from(Goldilocks::from_u64_reduced(7)); 18];
        let mut expected = None;
        let literal = allocation_counter::measure(|| {
            let mut table = zeros(1 << 18).unwrap();
            pad(&source.real, &shape, &mut table).unwrap();
            expected = Some(evaluate_base(&table, &point).unwrap());
        });
        let streamed = allocation_counter::measure(|| {
            assert_eq!(reader.extension(&point).unwrap(), expected.unwrap());
        });
        assert!(
            streamed.bytes_max < 32_768,
            "split weights peak {}",
            streamed.bytes_max
        );
        assert!(streamed.bytes_max * 32 < literal.bytes_max);
    }

    #[test]
    fn streamed_reads_refuse_malformed_witness_lengths_and_point_geometry() {
        let real = Source {
            real: vec![0; 2],
            padded: None,
        };
        assert!(matches!(
            TensorReader::new(&real, TensorId(0), &[3]),
            Err(StageError::Internal {
                rule: "a tensor's values fill its declared shape"
            })
        ));
        let padded = Source {
            real: vec![0; 3],
            padded: Some(vec![0; 3]),
        };
        assert!(matches!(
            TensorReader::new(&padded, TensorId(0), &[3]),
            Err(StageError::Internal {
                rule: "a tensor's values fill its declared shape"
            })
        ));
        let valid = Source {
            real: vec![0; 3],
            padded: None,
        };
        let reader = TensorReader::new(&valid, TensorId(0), &[3]).unwrap();
        assert!(matches!(
            reader.extension(&[GoldilocksExt2::ZERO]),
            Err(StageError::Internal {
                rule: "a claim's point has its tensor's coordinates"
            })
        ));
    }
}
