// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `D`'s weights as the weight commitment reads them (RFC-0040 'The matrices'): each weight
//! type's shape from the architecture (03 §2), and an [`IntegerModel`]'s tensors and channel
//! parameters, borrowed.
//!
//! The commitment's crate lays the matrices out from these and commits them:
//!
//! ```text
//! let layout = WeightLayout::new(claim_list.layout.weights, model.weight_shapes()?)?;
//! let commitment = WeightCommitment::commit(&layout, row_block, 0, &model.model_weights()?)?;
//! ```

use tessera_pcs::{
    ChannelParamsRef, ChannelValues, ModelWeights, WeightMatrix, WeightShape, WeightShapes,
    WeightTensorRef,
};

use crate::{ArchSpec, IntegerModel, IrError, QuantPoint, Site, WeightRef, WeightType};

impl WeightType {
    /// The weight commitment's matrix of this type, of the same declaration index (03 §2,
    /// RFC-0031).
    pub const fn matrix(self) -> WeightMatrix {
        match self {
            Self::Embed => WeightMatrix::Embed,
            Self::Q => WeightMatrix::Q,
            Self::K => WeightMatrix::K,
            Self::V => WeightMatrix::V,
            Self::O => WeightMatrix::O,
            Self::Gate => WeightMatrix::Gate,
            Self::Up => WeightMatrix::Up,
            Self::Down => WeightMatrix::Down,
            Self::LmHead => WeightMatrix::LmHead,
        }
    }
}

impl ArchSpec {
    /// Each weight type's shape, `Embed` to `LmHead`: output channels, its rows, by input
    /// channels (03 §2, RFC-0006 'Llama lowering'). With `q = n_heads·d_head` and
    /// `kv = n_kv_heads·d_head`: `Embed` is `d_model × vocab`, transposed; `Q` is `q × d_model`;
    /// `K` and `V` are `kv × d_model`; `O` is `d_model × q`; `Gate` and `Up` are
    /// `d_ff × d_model`; `Down` is `d_model × d_ff`; `LmHead` is `vocab × d_model`.
    ///
    /// # Errors
    ///
    /// [`IrError::NoLayers`] for a model of no layer, which RFC-0007 'Verifier validation'
    /// refuses (`n_layers ≥ 1`) and whose per-layer weights would have no layer count to pad
    /// (RFC-0040 'The matrices'); then [`IrError::Declaration`] when `q` or `kv` exceeds 32
    /// bits.
    pub fn weight_shapes(&self) -> Result<[WeightShape; 9], IrError> {
        // RFC-0007 'Verifier validation', architecture consistency: `n_layers ≥ 1`.
        if self.n_layers == 0 {
            return Err(IrError::NoLayers);
        }
        let wide = IrError::Declaration {
            rule: "n_heads · d_head and n_kv_heads · d_head fit in 32 bits",
        };
        let q = self.n_heads.checked_mul(self.d_head).ok_or(wide)?;
        let kv = self.n_kv_heads.checked_mul(self.d_head).ok_or(wide)?;
        let (d, ff, vocab) = (self.d_model, self.d_ff, self.vocab);
        Ok([
            (d, vocab),
            (q, d),
            (kv, d),
            (kv, d),
            (d, q),
            (ff, d),
            (ff, d),
            (d, ff),
            (vocab, d),
        ]
        .map(|(out, inputs)| WeightShape { out, inputs }))
    }
}

impl IntegerModel {
    /// What the weight layout follows from, all public: the architecture's shapes, and which
    /// weights have a `ChannelParams` (03 §2, RFC-0040 'The matrices').
    ///
    /// # Errors
    ///
    /// Those of [`ArchSpec::weight_shapes`].
    pub fn weight_shapes(&self) -> Result<WeightShapes, IrError> {
        Ok(WeightShapes {
            layers: self.arch.n_layers,
            shapes: self.arch.weight_shapes()?,
            channels: self
                .quant
                .channels
                .iter()
                .map(|params| (params.ty.matrix(), params.layer))
                .collect(),
        })
    }

    /// The weights and channel parameters, borrowed for the weight commitment, which checks that
    /// they are one per weight and of their shapes' lengths. Each value is checked against the
    /// domain `D` gives it, as the prover checks the hidden part before lowering (RFC-0006 'Llama
    /// lowering'), so the commitment never binds a value outside it.
    ///
    /// Each check reads every value of a tensor, whatever it holds: its control flow follows the
    /// shapes alone (INV-ARCH-2), and only a refusal says that some value broke it.
    ///
    /// # Errors
    ///
    /// Those of [`ArchSpec::weight_shapes`]; then, weight by weight in `D`'s order,
    /// [`IrError::Declaration`] for a weight whose rows and columns are not its type's output
    /// and input widths, and [`IrError::WeightOutOfRange`] for a weight with an entry of −128;
    /// then, `ChannelParams` by `ChannelParams`, [`IrError::Declaration`] for one that holds
    /// both class indices and multipliers, [`IrError::ClassOutOfRange`] for a class at or above
    /// its site's shift classes, and [`IrError::MultiplierOutOfRange`] for a multiplier of 0.
    pub fn model_weights(&self) -> Result<ModelWeights<'_>, IrError> {
        let shapes = self.arch.weight_shapes()?;
        let mut tensors = Vec::with_capacity(self.weights.len());
        for weight in &self.weights {
            let matrix = weight.ty.matrix();
            let shape = shapes[matrix.index()];
            if (weight.rows, weight.cols) != (shape.out, shape.inputs) {
                return Err(IrError::Declaration {
                    rule: "a weight's rows and columns are its type's output and input widths",
                });
            }
            // RFC-0005 'Pipeline' step 5, RFC-0006 'Range analysis': an INT8 weight lies in
            // [−127, 127], so −128 never occurs in D.
            if weight
                .data
                .iter()
                .fold(false, |found, &w| found | (w == i8::MIN))
            {
                return Err(IrError::WeightOutOfRange {
                    weight: WeightRef {
                        ty: weight.ty,
                        layer: weight.layer,
                    },
                });
            }
            tensors.push(WeightTensorRef {
                matrix,
                layer: weight.layer,
                data: &weight.data,
            });
        }
        let mut channels = Vec::with_capacity(self.quant.channels.len());
        for params in &self.quant.channels {
            let values = match (params.class.is_empty(), params.multiplier.is_empty()) {
                (false, false) => {
                    return Err(IrError::Declaration {
                        rule: "a ChannelParams holds class indices or multipliers, not both",
                    });
                }
                (true, false) => ChannelValues::Multipliers(&params.multiplier),
                _ => ChannelValues::Classes(&params.class),
            };
            let weight = WeightRef {
                ty: params.ty,
                layer: params.layer,
            };
            match values {
                // 03 §2 'Channels': a class indexes its site's shift classes.
                ChannelValues::Classes(classes) => {
                    let count = self.class_count(params.ty, params.layer);
                    if classes
                        .iter()
                        .fold(false, |found, &c| found | (usize::from(c) >= count))
                    {
                        return Err(IrError::ClassOutOfRange {
                            weight,
                            classes: count,
                        });
                    }
                }
                // 03 §2 'Channels', RFC-0025: a multiplier lies in [1, 2^16).
                ChannelValues::Multipliers(multipliers) => {
                    if multipliers.iter().fold(false, |found, &m| found | (m == 0)) {
                        return Err(IrError::MultiplierOutOfRange { weight });
                    }
                }
            }
            channels.push(ChannelParamsRef {
                matrix: params.ty.matrix(),
                layer: params.layer,
                values,
            });
        }
        Ok(ModelWeights { tensors, channels })
    }

    /// The shift classes of the site that the outputs of weight `(ty, layer)` reach, its point in
    /// `shift_class_sets` (03 §2 'Quantization'): `RopeQ` after `Q`, `RopeK` after `K`, `V` after
    /// `V`, `AttnResidual` after `O`, `Silu` after `Gate`, `Up` after `Up`, `MlpResidual` after
    /// `Down`. 0 when that site has no shift classes, an affine site or one the profile lacks,
    /// and for `Embed` and `LmHead`, which no rescale site follows.
    fn class_count(&self, ty: WeightType, layer: u32) -> usize {
        let site = match ty {
            WeightType::Q => Site::RopeQ,
            WeightType::K => Site::RopeK,
            WeightType::V => Site::V,
            WeightType::O => Site::AttnResidual,
            WeightType::Gate => Site::Silu,
            WeightType::Up => Site::Up,
            WeightType::Down => Site::MlpResidual,
            WeightType::Embed | WeightType::LmHead => return 0,
        };
        let point = QuantPoint::Layer { layer, site };
        self.quant
            .public
            .shift_class_sets
            .iter()
            .find(|(p, _)| *p == point)
            .map_or(0, |(_, shifts)| shifts.len())
    }
}
