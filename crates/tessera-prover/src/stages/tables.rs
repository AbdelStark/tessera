// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The tables a stage reads, from the witness: each tensor padded per dimension with zeros, a
//! stacked kind's layers one after another, its padded layers zero (RFC-0011 rule 1, RFC-0035,
//! RFC-0037 'Layer variables').

use tessera_codec::SecretBuf;
use tessera_compiler::schedule::{Kind, Layers};
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_ir::TensorId;

use super::{StageError, StageProver, WitnessSource};

impl<W: WitnessSource + ?Sized> StageProver<'_, W> {
    /// The variables of a tensor of `shape`, padded per dimension.
    pub(super) fn vars_of(shape: &[u32]) -> u32 {
        shape.iter().map(|&len| axis_vars(len)).sum()
    }

    /// The shape of `kind`'s tensors.
    pub(super) fn shape(&self, kind: Kind) -> Result<&[u32], StageError> {
        let tensor = self.tensor(kind, 0)?;
        Ok(&self.lowering.graph.tensors[tensor.0 as usize].shape)
    }

    /// The tensor of `kind` in `layer`.
    pub(super) fn tensor(&self, kind: Kind, layer: u32) -> Result<TensorId, StageError> {
        self.schedule
            .layers
            .tensor(&self.lowering.graph, kind, layer)
            .ok_or(StageError::Internal {
                rule: "a kind names a tensor in every layer",
            })
    }

    /// `kind`'s tensor in `layer`, padded per dimension with zeros.
    pub(super) fn slice(
        &self,
        kind: Kind,
        layer: u32,
    ) -> Result<SecretBuf<Goldilocks>, StageError> {
        let tensor = self.tensor(kind, layer)?;
        let shape = &self.lowering.graph.tensors[tensor.0 as usize].shape;
        let mut table = zeros(1usize << Self::vars_of(shape))?;
        let values = self.witness.values(tensor).ok_or(StageError::Internal {
            rule: "the witness holds every tensor of the graph",
        })?;
        pad(values, shape, &mut table)?;
        Ok(table)
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
            let values = self.witness.values(tensor).ok_or(StageError::Internal {
                rule: "the witness holds every tensor of the graph",
            })?;
            let at = layer as usize * block;
            pad(values, &shape, &mut table[at..at + block])?;
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
fn pad(values: &[i64], shape: &[u32], out: &mut [Goldilocks]) -> Result<(), StageError> {
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
