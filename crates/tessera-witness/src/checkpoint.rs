// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The checkpoint store (RFC-0018 'Witness engine'): a proof unit's layer-boundary residuals, from
//! which any layer's tensors regenerate without rerunning the layers before it.
//!
//! The residual stream after the embedding and after each layer, `c_0, …, c_L` (RFC-0037 'Layers
//! and kinds'), is kept for the unit's tokens, INT32: the lowering saturates every residual sum at
//! `residual_bits ≤ 31` bits (RFC-0022), and `c_0` lies inside the same range. [`Checkpoints::new`]
//! computes them as the witness pass writes them, one layer at a time, each from the last: it holds
//! the store and one layer's replay, never the whole witness.
//!
//! The residuals are secret-class activations (05 §3): each lives in a `SecretBuf`, allocated at its
//! length, which clears its memory when the store drops (OBS-6). The store lends them, and its debug
//! form shows its shapes.

use core::fmt;

use tessera_codec::{Secret, SecretBuf};
use tessera_compiler::Lowering;
use tessera_ir::{IntegerModel, TensorId};

use crate::replay::Replay;
use crate::structure::Structure;
use crate::{Engine, WitnessError};

/// The layer-boundary residuals `c_0, …, c_L` of one proof unit's tokens, `[n, d_model]` INT32 each
/// (RFC-0018 'Witness engine', RFC-0037).
///
/// The residuals are secret-class (05 §3), each in a `SecretBuf` that clears its memory when the
/// store drops (OBS-6). The store lends them, as slices, to the code that holds it, which keeps the
/// rules of [`Witness::tensor`](crate::Witness::tensor). The debug form shows the shapes, never a
/// value.
pub struct Checkpoints {
    /// The checkpoints' tensors in the lowering, `c_0` first.
    tensors: Vec<TensorId>,
    /// `n`, the unit's positions, and `d_model`, each checkpoint's width.
    positions: usize,
    width: usize,
    residuals: Vec<SecretBuf<i32>>,
}

/// Shows the number of checkpoints and their shape.
impl fmt::Debug for Checkpoints {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Checkpoints")
            .field("residuals", &self.residuals.len())
            .field("positions", &self.positions)
            .field("width", &self.width)
            .finish()
    }
}

impl Checkpoints {
    /// The checkpoints of `model` on `tokens`, on `engine`, the engine of `model`'s lowering for
    /// `tokens.len()` positions: `c_0` from the tokens, then each `c_{ℓ+1}` from `c_ℓ` alone, every
    /// value the first full pass's (RFC-0018 'Pass structure', P1). Two calls give the same
    /// residuals, whatever the engine's thread count, as a prover that reruns its phases after a
    /// crash requires (RFC-0018 'Crash consistency').
    ///
    /// # Errors
    ///
    /// As [`Engine::run`]: [`WitnessError::Usage`] for an engine made without a lowering, a
    /// sequence of another length or a token outside the vocabulary; [`WitnessError::Model`] for a
    /// model whose weights or classes the lowering cannot take; [`WitnessError::Range`], in debug
    /// builds, for a value outside its tensor's range; [`WitnessError::Memory`] when a tensor
    /// cannot be allocated; [`WitnessError::Internal`] for a residual outside INT32 or a lowering
    /// whose checkpoints are not its layers' boundaries.
    pub fn new(
        engine: &Engine<'_>,
        model: &IntegerModel,
        tokens: &Secret<Vec<u32>>,
    ) -> Result<Self, WitnessError> {
        let structure = Structure::new(engine, &model.arch)?;
        let replay = Replay::new(structure, model, tokens)?;
        let lowering = replay.lowering();
        let mut store = Self::empty(lowering)?;
        for &c in &lowering.checkpoints {
            let node = replay.structure.producer(c).ok_or(WitnessError::Internal {
                rule: "every checkpoint is a node's output",
            })?;
            let live = replay.compute(&store, &node.inputs)?;
            let mut outputs = {
                let operands = replay.operands(&store, node, &live)?;
                replay.run(node, &operands.all(&live)?)?
            };
            drop(live);
            let j = node.outputs.iter().position(|&t| t == c).unwrap_or(0);
            if j >= outputs.len() {
                return Err(WitnessError::Internal {
                    rule: "every checkpoint is a node's output",
                });
            }
            let residual = outputs.swap_remove(j);
            drop(outputs);
            store.residuals.push(int32(c, &residual)?);
        }
        Ok(store)
    }

    /// An empty store for `lowering`'s checkpoints, which are `[n, d_model]` each.
    fn empty(lowering: &Lowering) -> Result<Self, WitnessError> {
        let shape = |t: TensorId| {
            lowering
                .graph
                .tensors
                .get(t.0 as usize)
                .map(|s| s.shape.as_slice())
        };
        let first = lowering.checkpoints.first().copied().and_then(shape);
        let Some(&[positions, width]) = first else {
            return Err(WitnessError::Internal {
                rule: "the checkpoints are [n, d_model] residuals",
            });
        };
        if lowering
            .checkpoints
            .iter()
            .any(|&c| shape(c) != Some(&[positions, width][..]))
        {
            return Err(WitnessError::Internal {
                rule: "the checkpoints are [n, d_model] residuals",
            });
        }
        Ok(Self {
            tensors: lowering.checkpoints.clone(),
            positions: positions as usize,
            width: width as usize,
            residuals: Vec::with_capacity(lowering.checkpoints.len()),
        })
    }

    /// Whether the store holds `lowering`'s checkpoints: its tensors and shapes, each one whole.
    pub(crate) fn fits(&self, lowering: &Lowering) -> bool {
        Self::empty(lowering).is_ok_and(|empty| {
            empty.tensors == self.tensors
                && (empty.positions, empty.width) == (self.positions, self.width)
                && self.residuals.len() == self.tensors.len()
                && self
                    .residuals
                    .iter()
                    .all(|r| r.len() == self.positions * self.width)
        })
    }

    /// The number of checkpoints, `L + 1`.
    pub fn len(&self) -> usize {
        self.residuals.len()
    }

    /// Whether the store holds no checkpoint: never, for a store [`Checkpoints::new`] returns.
    pub fn is_empty(&self) -> bool {
        self.residuals.is_empty()
    }

    /// Checkpoint `c_k`, `[n, d_model]`, flat and row-major, lent to the code that computes on it;
    /// `None` past `c_L`.
    pub fn residual(&self, k: usize) -> Option<&[i32]> {
        self.residuals.get(k).map(|r| &r[..])
    }

    /// The bytes the residuals take: `(L + 1)·n·d_model` INT32 values.
    pub fn bytes(&self) -> usize {
        self.residuals
            .iter()
            .map(SecretBuf::capacity)
            .sum::<usize>()
            * size_of::<i32>()
    }
}

/// `values` as INT32, in a buffer of exactly their length.
fn int32(tensor: TensorId, values: &[i64]) -> Result<SecretBuf<i32>, WitnessError> {
    let mut out =
        SecretBuf::try_with_capacity(values.len()).map_err(|_| WitnessError::Memory { tensor })?;
    let mut fits = true;
    out.extend(values.iter().map(|&v| {
        let narrow = i32::try_from(v);
        fits &= narrow.is_ok();
        narrow.unwrap_or(0)
    }));
    match fits {
        true => Ok(out),
        false => Err(WitnessError::Internal {
            rule: "a layer-boundary residual fits INT32 (RFC-0018 'Witness engine')",
        }),
    }
}
