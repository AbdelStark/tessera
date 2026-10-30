// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Borrowed integer tensors shared by tile and reduction consumers (RFC-0018).

use core::fmt;

use tessera_codec::{Secret, SecretBuf};
use tessera_ir::{IntegerModel, TensorId};

use crate::engine::Input;
use crate::replay::Replay;
use crate::structure::Structure;
use crate::{Checkpoints, Engine, WitnessError};

/// A model's integer tensors, flat and row-major, in clearing or borrowed model storage.
/// Values are secret-class (05 §3): consumers borrow them for computation, never logs or
/// errors. Model weights may remain in the caller's INT8 allocation instead of being widened.
pub trait TensorSource {
    /// Real values of `tensor`, or `None` when this source has no `i64` representation.
    fn values(&self, tensor: TensorId) -> Option<&[i64]>;

    /// Borrowed INT8 model weights, used when [`Self::values`] returns `None`.
    /// Other graph inputs and activations remain `i64`. The default supplies no INT8 weights.
    fn int8_values(&self, tensor: TensorId) -> Option<&[i8]> {
        let _ = tensor;
        None
    }

    /// Values over the padded shape, each axis padded to a power of two, with the last axis's
    /// index in the lowest bits (RFC-0011 rule 1). A consumer prefers this representation when
    /// it accepts explicitly computed padding (RFC-0072 'Padded positions').
    ///
    /// The default supplies real positions only, whose other positions are zero. A source
    /// whose lookup is nonzero at padding must instead supply its computed padded values;
    /// `tessera_prover::PaddedWitness` can derive them from an integer source.
    fn padded(&self, tensor: TensorId) -> Option<&[i64]> {
        let _ = tensor;
        None
    }
}

impl TensorSource for crate::Witness {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        self.tensor(tensor).map(|values| &values[..])
    }
}

/// Integer tensors replayed from checkpoints, retaining nonweight values while borrowing every
/// model matrix in its original INT8 allocation (RFC-0018 P3). No weight is widened or copied.
///
/// This cache holds *all* nonweight tensors, including full scores. It is useful when that
/// public working set fits; it does not claim bounded memory independent of context length.
/// The caller separately owns the model, engine, tokens and checkpoint store used to build it.
/// Every retained derived tensor lives in a clearing buffer (OBS-6).
pub struct CheckpointTensors<'model> {
    values: Vec<Option<SecretBuf<i64>>>,
    weights: Vec<Option<&'model [i8]>>,
}

impl fmt::Debug for CheckpointTensors<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckpointTensors")
            .field("tensors", &self.values.len())
            .field("borrowed_weights", &self.weights.iter().flatten().count())
            .finish_non_exhaustive()
    }
}

impl<'model> CheckpointTensors<'model> {
    /// Replay all nonweight tensors in one cone and transfer their clearing allocations.
    /// The store must belong to the same lowering, model and canonical token sequence.
    /// Weights are read through the engine's existing validated reader; the cache only borrows
    /// them. The returned value borrows no engine, token buffer, lowering or checkpoint store.
    ///
    /// # Errors
    /// As [`crate::Regenerator::new`] for inputs and checkpoint geometry, or `Memory` when
    /// metadata/tensor allocations fail. Missing replay outputs return `Internal`.
    pub fn new(
        engine: &Engine<'_>,
        model: &'model IntegerModel,
        tokens: &Secret<Vec<u32>>,
        checkpoints: &Checkpoints,
    ) -> Result<Self, WitnessError> {
        let structure = Structure::new(engine, &model.arch)?;
        let replay = Replay::new(structure, model, tokens)?;
        let lowering = replay.lowering();
        if !checkpoints.fits(lowering) {
            return Err(WitnessError::Usage {
                rule: "the checkpoints are the lowering's, for its sequence",
            });
        }
        let count = lowering.graph.tensors.len();
        let memory = WitnessError::Memory {
            tensor: lowering.tokens,
        };
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|_| memory)?;
        values.resize_with(count, || None);
        let mut weights = Vec::new();
        weights.try_reserve_exact(count).map_err(|_| memory)?;
        weights.resize(count, None);
        for &(_, tensor) in &lowering.graph.weights {
            let Input::Weight(entries) = engine.graph_input(lowering, model, tokens, tensor)?
            else {
                return Err(WitnessError::Internal {
                    rule: "a model weight is borrowed from its original INT8 matrix",
                });
            };
            weights[tensor.0 as usize] = Some(entries);
        }
        let mut requested = Vec::new();
        requested.try_reserve_exact(count).map_err(|_| memory)?;
        requested.extend(
            lowering
                .graph
                .tensors
                .iter()
                .filter(|tensor| weights[tensor.id.0 as usize].is_none())
                .map(|tensor| tensor.id),
        );
        // Asking for every nonweight output intentionally materializes scores too. The full
        // public nonweight domain, checkpoint copies and replay scratch must be budgeted.
        let mut live = replay.compute(checkpoints, &requested)?;
        for tensor in requested {
            let value = if replay.structure.computed(tensor) {
                live.take(tensor).ok_or(WitnessError::Internal {
                    rule: "a tensor replay retains its requested output",
                })?
            } else {
                let Input::Values(values) = replay.leaf(checkpoints, tensor)? else {
                    return Err(WitnessError::Internal {
                        rule: "nonweight cache inputs have an integer representation",
                    });
                };
                values
            };
            values[tensor.0 as usize] = Some(value);
        }
        Ok(Self { values, weights })
    }

    /// Retained storage, including containers and descriptors, excluding all caller-owned
    /// model/checkpoint/engine allocations and constructor scratch. Overflow returns `None`.
    pub fn storage_bytes(&self) -> Option<usize> {
        self.values.iter().flatten().try_fold(
            size_of::<Self>()
                .checked_add(
                    self.values
                        .capacity()
                        .checked_mul(size_of::<Option<SecretBuf<i64>>>())?,
                )?
                .checked_add(
                    self.weights
                        .capacity()
                        .checked_mul(size_of::<Option<&[i8]>>())?,
                )?,
            |bytes, values| bytes.checked_add(values.capacity().checked_mul(size_of::<i64>())?),
        )
    }
}

impl TensorSource for CheckpointTensors<'_> {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        self.values.get(tensor.0 as usize)?.as_deref()
    }

    fn int8_values(&self, tensor: TensorId) -> Option<&[i8]> {
        self.weights.get(tensor.0 as usize).copied().flatten()
    }
}
