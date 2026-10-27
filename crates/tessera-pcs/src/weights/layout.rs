// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The ten matrices of the weight commitment (RFC-0040 'The matrices', RFC-0012 'Layout'): every
//! entry's position, the committed rows, and the rows' values.
//!
//! # The nine weight types
//!
//! A weight of `out × in` (rows are output channels) pads `out`, `in` and, for the seven
//! per-layer types, `n_layers` to powers of two, `out_pad`, `in_pad` and `L_pad`; `Embed` and
//! `LmHead` have one layer. Its message width is `k_T = max(in_pad, min(k_W, out_pad · in_pad))`,
//! `a_T = log2(k_T / in_pad)` and `h_T = log2 out_pad − a_T`. Entry `W_ℓ[o][i]` lies at row
//! `ℓ·2^{h_T} + ⌊o / 2^{a_T}⌋` and column `(o mod 2^{a_T})·in_pad + i`. A row is structurally
//! zero, and skipped, when `ℓ ≥ n_layers` or `⌊o / 2^{a_T}⌋·2^{a_T} ≥ out`; the others are
//! committed, in increasing order, `m_T = n_layers·⌈out / 2^{a_T}⌉` of them.
//!
//! # The channel parameters
//!
//! The tensor `[16, L_pad, C]` is flattened row-major: entry `f = (t·L_pad + ℓ)·C + c`, `t` the
//! weight type's index (`Embed` 0 to `LmHead` 8), lies at row `⌊f / k_CP⌋` and column
//! `f mod k_CP`, with `k_CP = min(k_W, 16·L_pad·C)`. `C` is the largest padded output width among
//! the weights that have a `ChannelParams`, 1 when none has. A position is live when its weight
//! has a `ChannelParams` and `c` is below the weight's output width; its entry is the channel's
//! class index or multiplier. A row is committed when it holds a live position.
//!
//! Every other entry, padding, is 0, and a weight is its true integer in the centered
//! representation (RFC-0002): `w ≥ 0` is `w`, `w < 0` is `P − |w|`.
//!
//! # Rows
//!
//! One function lays out one committed row, and both readers of the weights use it: the
//! commitment, which writes every row ([`WeightLayout::write_rows`]), and the fold pass, which
//! reads one row at a time ([`WeightLayout::rows`], [`WeightRows`]).
//!
//! # Secrets
//!
//! The layout is public: it follows from the shapes and from which weights have a
//! `ChannelParams`, both of which the claim list shows (03 §2). The weights and channel parameters
//! are secret-class (05 §3): [`ModelWeights`] borrows them from `D`, the layout reads each once
//! into the buffer the caller gives, and the `Debug` forms show shapes and counts only. Control
//! flow depends on the shapes, never on a value (INV-ARCH-2): each entry is written whatever it
//! holds, and a weight is embedded by the branch-free `from_i64`.

use core::fmt;

use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};

use crate::{MatrixRows, PcsError, WeightLayoutParams, WeightMatrix};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

/// The `ChannelParams` tensor's types: the nine weight types, padded to 16 (RFC-0012 'Layout').
const LOG2_CHANNEL_TYPES: u32 = 4;

/// A weight's shape (03 §2): `out` output channels, its rows, by `inputs` input channels, its
/// columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightShape {
    /// The output channels: the weight's rows.
    pub out: u32,
    /// The input channels: the weight's columns.
    pub inputs: u32,
}

/// What the weights' layout follows from, all public (RFC-0040 'The matrices'): the
/// architecture's shapes, and which weights have a `ChannelParams`, which the claim list shows
/// (03 §2).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightShapes {
    /// `n_layers`: the layers of each per-layer type, `Q` to `Down`.
    pub layers: u32,
    /// Each weight type's shape, in [`WeightMatrix`] order from `Embed` to `LmHead`.
    pub shapes: [WeightShape; 9],
    /// The weights that have a `ChannelParams`, by type and layer (0 for `Embed` and `LmHead`),
    /// in any order.
    pub channels: Vec<(WeightMatrix, u32)>,
}

/// The kind of a matrix, with what its layout depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatrixKind {
    /// A weight type: `W_ℓ[o][i]` at row `ℓ·2^h + ⌊o / 2^a⌋` and column
    /// `(o mod 2^a)·in_pad + i`.
    Weight {
        /// The weight's shape.
        shape: WeightShape,
        /// Its layers: `n_layers` for `Q` to `Down`, 1 for `Embed` and `LmHead`.
        layers: u32,
        /// `a_T`, the bits of the output index in a column: `k_T = 2^a_T · in_pad`.
        a: u32,
        /// `h_T`, the bits of the output index in a row: `log2 out_pad − a_T`.
        h: u32,
    },
    /// The channel parameters: entry `(t·L_pad + ℓ)·C + c` at row `⌊f / k_CP⌋` and column
    /// `f mod k_CP`.
    ChannelParams {
        /// `C`, the largest padded output width among the weights that have a `ChannelParams`,
        /// 1 when none has.
        channel_width: u64,
    },
}

/// One matrix's layout: its message width `k_T`, the bits of its row index, its committed rows
/// `m_T`, and how its entries are placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixLayout {
    matrix: WeightMatrix,
    kind: MatrixKind,
    log2_k: u32,
    row_vars: u32,
    rows: u64,
}

impl MatrixLayout {
    /// The matrix.
    pub fn matrix(&self) -> WeightMatrix {
        self.matrix
    }

    /// How its entries are placed.
    pub fn kind(&self) -> MatrixKind {
        self.kind
    }

    /// `log2 k_T`: a row holds `k_T` values.
    pub fn log2_k(&self) -> u32 {
        self.log2_k
    }

    /// `k_T`, the values in a row.
    pub fn message_len(&self) -> usize {
        1 << self.log2_k
    }

    /// The bits of the row index: the matrix has `2^row_vars` rows, committed or structurally
    /// zero.
    pub fn row_vars(&self) -> u32 {
        self.row_vars
    }

    /// `m_T`, the committed rows.
    pub fn rows(&self) -> u64 {
        self.rows
    }
}

/// The layout of all ten matrices, from the parameter set's [`WeightLayoutParams`] and the
/// public [`WeightShapes`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightLayout {
    params: WeightLayoutParams,
    shapes: WeightShapes,
    /// The weights that have a `ChannelParams`, sorted by `(t, ℓ)`.
    channels: Vec<(WeightMatrix, u32)>,
    matrices: [MatrixLayout; 10],
}

impl WeightLayout {
    /// The layout of every matrix (RFC-0040 'The matrices').
    ///
    /// # Errors
    ///
    /// In [`WeightMatrix`] order, for the first matrix that fails: [`PcsError::EmptyWeight`]
    /// for a weight type with no output or no input, [`PcsError::InverseRate`] if the layout's
    /// inverse rate is `2^0`, then [`PcsError::TooLong`] if its codewords,
    /// `2^log2_inv_rate·k_T` symbols, would exceed `2^32`. Then, for the weights that have a
    /// `ChannelParams`, in their order:
    /// [`PcsError::WeightPlace`] for one that is not a weight type of a layer it has, and
    /// [`PcsError::WeightTwice`] for one listed twice; and [`PcsError::RowVars`] if the
    /// `ChannelParams` matrix would have more than `2^64` rows.
    pub fn new(params: WeightLayoutParams, shapes: WeightShapes) -> Result<Self, PcsError> {
        let mut matrices = [MatrixLayout {
            matrix: WeightMatrix::ChannelParams,
            kind: MatrixKind::ChannelParams { channel_width: 1 },
            log2_k: 0,
            row_vars: 0,
            rows: 0,
        }; 10];
        for matrix in WeightMatrix::WEIGHTS {
            let shape = shapes.shapes[matrix.index()];
            if shape.out == 0 || shape.inputs == 0 {
                return Err(PcsError::EmptyWeight { matrix });
            }
            let layers = if matrix.per_layer() { shapes.layers } else { 1 };
            let (log2_out, log2_in) = (log2_pad(shape.out), log2_pad(shape.inputs));
            let log2_k = log2_in.max(u32::from(params.log2_k).min(log2_out + log2_in));
            check_code(&params, log2_k)?;
            let a = log2_k - log2_in;
            let h = log2_out - a;
            // ⌈out / 2^a⌉ rows per layer hold an output; layers·that < 2^64.
            let per_layer = (u64::from(shape.out) + (1 << a) - 1) >> a;
            matrices[matrix.index()] = MatrixLayout {
                matrix,
                kind: MatrixKind::Weight {
                    shape,
                    layers,
                    a,
                    h,
                },
                log2_k,
                row_vars: log2_pad(layers) + h,
                rows: u64::from(layers) * per_layer,
            };
        }
        let mut channels = shapes.channels.clone();
        for &(matrix, layer) in &channels {
            if !has_weight(&shapes, matrix, layer) {
                return Err(PcsError::WeightPlace {
                    matrix,
                    layer,
                    channels: true,
                });
            }
        }
        channels.sort_unstable();
        if let Some(pair) = channels.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(PcsError::WeightTwice {
                matrix: pair[0].0,
                layer: pair[0].1,
                channels: true,
            });
        }
        let log2_c = channels
            .iter()
            .map(|&(matrix, _)| log2_pad(shapes.shapes[matrix.index()].out))
            .max()
            .unwrap_or(0);
        let log2_total = LOG2_CHANNEL_TYPES + log2_pad(shapes.layers) + log2_c;
        let log2_k = u32::from(params.log2_k).min(log2_total);
        check_code(&params, log2_k)?;
        let row_vars = log2_total - log2_k;
        if row_vars > u64::BITS {
            return Err(PcsError::RowVars { row_vars });
        }
        let mut layout = Self {
            params,
            shapes,
            channels,
            matrices,
        };
        let rows = layout
            .channel_row_ranges(log2_c, log2_k)
            .map(|(first, last)| last - first + 1)
            .sum();
        layout.matrices[WeightMatrix::ChannelParams.index()] = MatrixLayout {
            matrix: WeightMatrix::ChannelParams,
            kind: MatrixKind::ChannelParams {
                channel_width: 1 << log2_c,
            },
            log2_k,
            row_vars,
            rows,
        };
        Ok(layout)
    }

    /// The parameter set's layout parameters.
    pub fn params(&self) -> &WeightLayoutParams {
        &self.params
    }

    /// The shapes the layout follows from.
    pub fn shapes(&self) -> &WeightShapes {
        &self.shapes
    }

    /// The layout of `matrix`.
    pub fn matrix(&self, matrix: WeightMatrix) -> &MatrixLayout {
        &self.matrices[matrix.index()]
    }

    /// Checks that this layout is sound for `tessera-protocol/2`'s masked weight commitments
    /// (RFC-0012 'Layout', RFC-0007 step 6, #564): `P_W > 0`, `k_W + P_W < n_W`, and every
    /// committed matrix (`m_T > 0`) satisfies `k_T ≥ k_W` (`(k_T + P_W) / n_T ≤ (k_W + P_W) / n_W`).
    ///
    /// A `ChannelParams` matrix with `m_CP == 0` commits no row, has `B_CP = 0`, and is never
    /// opened, so its nominal `k_CP` is exempt.
    ///
    /// # Errors
    ///
    /// - [`PcsError::NoMask`] if `params.mask_coeffs == 0`;
    /// - [`PcsError::InverseRate`] or [`PcsError::TooLong`] if the nominal target code `(k_W, n_W)`
    ///   is invalid;
    /// - [`PcsError::MaskedWeightRate`] if `k_W + P_W ≥ n_W`;
    /// - [`PcsError::NarrowWeight`] for the first committed matrix in [`WeightMatrix`] order whose
    ///   message width `k_T` is below `k_W`.
    pub fn check_masked(&self) -> Result<(), PcsError> {
        if self.params.mask_coeffs == 0 {
            return Err(PcsError::NoMask);
        }
        let min_log2_k = u32::from(self.params.log2_k);
        check_code(&self.params, min_log2_k)?;
        let k_w = 1u64 << min_log2_k;
        let n_w = k_w << u32::from(self.params.log2_inv_rate);
        let p_w = u64::from(self.params.mask_coeffs);
        if k_w + p_w >= n_w {
            return Err(PcsError::MaskedWeightRate {
                log2_k: self.params.log2_k,
                log2_inv_rate: self.params.log2_inv_rate,
                mask_coeffs: self.params.mask_coeffs,
            });
        }
        for matrix in WeightMatrix::ALL {
            let shape = &self.matrices[matrix.index()];
            if shape.rows > 0 && shape.log2_k < min_log2_k {
                return Err(PcsError::NarrowWeight {
                    matrix,
                    log2_k: shape.log2_k,
                    min_log2_k,
                });
            }
        }
        Ok(())
    }

    /// Maps a reading site's claim on weight type `matrix`'s channel parameters at `channel_point`
    /// (`log2 out_pad` coordinates) and `layer_point` (`log2 L_pad` coordinates for per-layer
    /// weights, empty for `Embed` and `LmHead`) to its full point on `ChannelParams` (RFC-0052
    /// 'Point on `Ã_CP`'): `channel_point ‖ 0^{γ − c_i} ‖ layer_point ‖ vertex_4(matrix)`.
    ///
    /// Splitting the returned point at `self.matrix(WeightMatrix::ChannelParams).log2_k()` gives
    /// `(column_point, row_point)`.
    ///
    /// # Errors
    ///
    /// [`PcsError::WeightPlace`] if `matrix` has no `ChannelParams` in this layout,
    /// [`PcsError::ClaimPoint`] if `channel_point` does not have `log2 out_pad` coordinates, and
    /// [`PcsError::LayerPoint`] if `layer_point` does not have `log2 L_pad` coordinates (or 0 for
    /// `Embed` and `LmHead`).
    pub fn channel_point(
        &self,
        matrix: WeightMatrix,
        channel_point: &[Fp2],
        layer_point: &[Fp2],
    ) -> Result<Vec<Fp2>, PcsError> {
        if !self.channels.iter().any(|&(m, _)| m == matrix) {
            return Err(PcsError::WeightPlace {
                matrix,
                layer: 0,
                channels: true,
            });
        }
        let c_i = log2_pad(self.shapes.shapes[matrix.index()].out) as usize;
        if channel_point.len() != c_i {
            return Err(PcsError::ClaimPoint {
                expected: c_i,
                found: channel_point.len(),
            });
        }
        let log2_l_pad = log2_pad(self.shapes.layers) as usize;
        let expected_layers = if matrix.per_layer() { log2_l_pad } else { 0 };
        if layer_point.len() != expected_layers {
            return Err(PcsError::LayerPoint {
                matrix,
                expected: expected_layers,
                found: layer_point.len(),
            });
        }
        let MatrixKind::ChannelParams { channel_width } =
            self.matrices[WeightMatrix::ChannelParams.index()].kind
        else {
            unreachable!();
        };
        let gamma = channel_width.trailing_zeros() as usize;
        let mut point = Vec::with_capacity(gamma + log2_l_pad + LOG2_CHANNEL_TYPES as usize);
        point.extend_from_slice(channel_point);
        point.resize(gamma, Fp2::ZERO);
        point.extend_from_slice(layer_point);
        point.resize(gamma + log2_l_pad, Fp2::ZERO);
        let tag = matrix.index();
        for bit in 0..LOG2_CHANNEL_TYPES {
            point.push(if (tag >> bit) & 1 == 1 {
                Fp2::ONE
            } else {
                Fp2::ZERO
            });
        }
        Ok(point)
    }

    /// The committed rows of `matrix`, in increasing order: `m_T` of them.
    pub fn committed_rows(&self, matrix: WeightMatrix) -> impl Iterator<Item = u64> + '_ {
        let layout = self.matrices[matrix.index()];
        let (weights, channels) = match layout.kind {
            MatrixKind::Weight { layers, h, .. } => {
                let per_layer = layout.rows.checked_div(u64::from(layers)).unwrap_or(0);
                let rows = (0..u64::from(layers))
                    .flat_map(move |layer| (0..per_layer).map(move |o_hi| (layer << h) + o_hi));
                (Some(rows), None)
            }
            MatrixKind::ChannelParams { channel_width } => {
                let ranges = self.channel_row_ranges(channel_width.trailing_zeros(), layout.log2_k);
                (None, Some(ranges.flat_map(|(first, last)| first..=last)))
            }
        };
        weights
            .into_iter()
            .flatten()
            .chain(channels.into_iter().flatten())
    }

    /// The committed rows of `matrix`, `m_T` rows of `k_T` values in increasing row order,
    /// written over `out`, whatever it held. The values are secret-class: `out` belongs in a
    /// `Secret` or a `SecretBuf`, as [`WeightCommitment::commit`](crate::WeightCommitment::commit)
    /// keeps them (05 §3 OBS-6).
    ///
    /// # Errors
    ///
    /// Those of [`ModelWeights`]' checks ([`PcsError::WeightPlace`], [`PcsError::WeightLength`],
    /// [`PcsError::WeightTwice`], [`PcsError::WeightMissing`]), then
    /// [`PcsError::MatrixLength`] unless `out` holds `m_T·k_T` values. Nothing is written on
    /// error.
    pub fn write_rows(
        &self,
        matrix: WeightMatrix,
        weights: &ModelWeights<'_>,
        out: &mut [Fp],
    ) -> Result<(), PcsError> {
        let checked = self.check(weights)?;
        let layout = &self.matrices[matrix.index()];
        let expected = usize::try_from(layout.rows)
            .ok()
            .and_then(|rows| rows.checked_mul(layout.message_len()));
        if expected != Some(out.len()) {
            return Err(PcsError::MatrixLength {
                rows: usize::try_from(layout.rows).unwrap_or(usize::MAX),
                row_len: layout.message_len(),
                found: out.len(),
            });
        }
        self.fill(matrix, &checked, out);
        Ok(())
    }

    /// Checks `weights` against the shapes, and returns its tensors in canonical order, `Embed`,
    /// `Q` to `Down` layer by layer, `LmHead`, and its channel parameters in the order of
    /// `self.channels`.
    pub(crate) fn check<'a>(&self, weights: &ModelWeights<'a>) -> Result<Checked<'a>, PcsError> {
        let mut tensors: Vec<(WeightMatrix, u32, &'a [i8])> =
            Vec::with_capacity(weights.tensors.len());
        for tensor in &weights.tensors {
            let (matrix, layer) = (tensor.matrix, tensor.layer);
            if !self.is_weight(matrix, layer) {
                return Err(PcsError::WeightPlace {
                    matrix,
                    layer,
                    channels: false,
                });
            }
            let shape = self.shapes.shapes[matrix.index()];
            let expected = u64::from(shape.out) * u64::from(shape.inputs);
            if tensor.data.len() as u64 != expected {
                return Err(PcsError::WeightLength {
                    matrix,
                    layer,
                    channels: false,
                    expected: usize::try_from(expected).unwrap_or(usize::MAX),
                    found: tensor.data.len(),
                });
            }
            tensors.push((matrix, layer, tensor.data));
        }
        tensors.sort_unstable_by_key(|&(matrix, layer, _)| (matrix, layer));
        if let Some(pair) = tensors
            .windows(2)
            .find(|pair| (pair[0].0, pair[0].1) == (pair[1].0, pair[1].1))
        {
            return Err(PcsError::WeightTwice {
                matrix: pair[0].0,
                layer: pair[0].1,
                channels: false,
            });
        }
        // Every tensor has a place and none is repeated, so the sorted tensors are the canonical
        // sequence with some missing; the first missing is where the two first differ.
        let canonical = self.weights_in_order();
        let found = tensors.iter().map(|&(matrix, layer, _)| (matrix, layer));
        if let Some((matrix, layer)) = canonical
            .zip(found.map(Some).chain(core::iter::repeat(None)))
            .find(|&(expected, found)| found != Some(expected))
            .map(|(expected, _)| expected)
        {
            return Err(PcsError::WeightMissing {
                matrix,
                layer,
                channels: false,
            });
        }

        let mut channels: Vec<(WeightMatrix, u32, ChannelValues<'a>)> =
            Vec::with_capacity(weights.channels.len());
        for params in &weights.channels {
            let (matrix, layer) = (params.matrix, params.layer);
            if self.channels.binary_search(&(matrix, layer)).is_err() {
                return Err(PcsError::WeightPlace {
                    matrix,
                    layer,
                    channels: true,
                });
            }
            let expected = self.shapes.shapes[matrix.index()].out;
            let found = params.values.len();
            if found as u64 != u64::from(expected) {
                return Err(PcsError::WeightLength {
                    matrix,
                    layer,
                    channels: true,
                    expected: usize::try_from(expected).unwrap_or(usize::MAX),
                    found,
                });
            }
            channels.push((matrix, layer, params.values));
        }
        channels.sort_unstable_by_key(|&(matrix, layer, _)| (matrix, layer));
        if let Some(pair) = channels
            .windows(2)
            .find(|pair| (pair[0].0, pair[0].1) == (pair[1].0, pair[1].1))
        {
            return Err(PcsError::WeightTwice {
                matrix: pair[0].0,
                layer: pair[0].1,
                channels: true,
            });
        }
        if let Some(&(matrix, layer)) = self
            .channels
            .iter()
            .zip(channels.iter().map(Some).chain(core::iter::repeat(None)))
            .find(|&(&(m, l), found)| found.is_none_or(|&(fm, fl, _)| (fm, fl) != (m, l)))
            .map(|(expected, _)| expected)
        {
            return Err(PcsError::WeightMissing {
                matrix,
                layer,
                channels: true,
            });
        }
        Ok(Checked {
            tensors: tensors.into_iter().map(|(_, _, data)| data).collect(),
            channels: channels.into_iter().map(|(_, _, values)| values).collect(),
        })
    }

    /// Writes the committed rows of `matrix` over `out`, which holds `m_T·k_T` values, row by
    /// row with [`fill_row`](Self::fill_row): the commitment and the fold pass place every entry
    /// with one code.
    pub(crate) fn fill(&self, matrix: WeightMatrix, checked: &Checked<'_>, out: &mut [Fp]) {
        let k = self.matrices[matrix.index()].message_len();
        for ((index, row), values) in self
            .committed_rows(matrix)
            .enumerate()
            .zip(out.chunks_exact_mut(k))
        {
            self.fill_row(matrix, checked, row, index, values);
        }
    }

    /// Writes committed row `index` of `matrix`, the `index`-th in increasing order and row
    /// `row` of the matrix, over `out`, which holds `k_T` values: every entry, padding as 0.
    pub(crate) fn fill_row(
        &self,
        matrix: WeightMatrix,
        checked: &Checked<'_>,
        row: u64,
        index: usize,
        out: &mut [Fp],
    ) {
        out.fill(Fp::ZERO);
        let layout = &self.matrices[matrix.index()];
        let k = layout.message_len();
        match layout.kind {
            MatrixKind::Weight {
                shape, layers, a, ..
            } => {
                // Each layer commits ⌈out / 2^a⌉ rows, m_T / layers; row o_hi of a layer holds
                // outputs o_hi·2^a to o_hi·2^a + 2^a − 1, those below `out`, each an input row
                // at column (o mod 2^a)·in_pad. `in ≥ 1` and k_T ≤ 2^31: the slices fit.
                let per_layer = (layout.rows / u64::from(layers)) as usize;
                let (layer, o_hi) = (index / per_layer, index % per_layer);
                let (inputs, outputs) = (shape.inputs as usize, shape.out as usize);
                let in_pad = k >> a;
                let data = checked.tensors[self.first_tensor(matrix) + layer];
                let first = o_hi << a;
                let last = ((o_hi + 1) << a).min(outputs);
                for (o_lo, weights) in data[first * inputs..last * inputs]
                    .chunks_exact(inputs)
                    .enumerate()
                {
                    for (entry, &w) in out[o_lo * in_pad..][..inputs].iter_mut().zip(weights) {
                        *entry = Fp::from_i64(i64::from(w));
                    }
                }
            }
            MatrixKind::ChannelParams { channel_width } => {
                // The row holds the flattened indices [r·k, (r + 1)·k): each weight's live
                // positions [base, base + out) that fall in it.
                let log2_c = channel_width.trailing_zeros();
                let log2_l_pad = log2_pad(self.shapes.layers);
                let start = u128::from(row) << layout.log2_k;
                let end = start + k as u128;
                for (&(matrix, layer), values) in self.channels.iter().zip(&checked.channels) {
                    let base = channel_base(matrix, layer, log2_l_pad, log2_c);
                    let (low, high) = (base.max(start), (base + values.len() as u128).min(end));
                    for f in low..high {
                        out[(f - start) as usize] = values.get((f - base) as usize);
                    }
                }
            }
        }
    }

    /// A reader of the committed rows of `matrix`, laid out from `weights` one row at a time
    /// ([`WeightRows`]), for the fold pass ([`WeightFold`](crate::WeightFold)).
    ///
    /// # Errors
    ///
    /// Those of [`ModelWeights`]' checks, as [`write_rows`](Self::write_rows) has them
    /// ([`PcsError::WeightPlace`], [`PcsError::WeightLength`], [`PcsError::WeightTwice`],
    /// [`PcsError::WeightMissing`]): every weight of `D` is checked, whichever matrix is read.
    pub fn rows<'w>(
        &self,
        matrix: WeightMatrix,
        weights: &ModelWeights<'w>,
    ) -> Result<WeightRows<'_, 'w>, PcsError> {
        let checked = self.check(weights)?;
        let rows = match self.matrices[matrix.index()].kind {
            MatrixKind::Weight { .. } => Vec::new(),
            MatrixKind::ChannelParams { .. } => self.committed_rows(matrix).collect(),
        };
        Ok(WeightRows {
            layout: self,
            matrix,
            checked,
            rows,
        })
    }

    /// The merged ranges `(first, last)` of the rows that hold a live position of the
    /// `ChannelParams` matrix, in increasing order. The live positions of one weight are
    /// `[base, base + out)` with `base = (t·L_pad + ℓ)·C`; they increase with `(t, ℓ)` and never
    /// overlap, so two consecutive weights share at most one row.
    fn channel_row_ranges(
        &self,
        log2_c: u32,
        log2_k: u32,
    ) -> impl Iterator<Item = (u64, u64)> + '_ {
        let log2_l_pad = log2_pad(self.shapes.layers);
        let mut merged: Option<(u64, u64)> = None;
        let mut ranges = self.channels.iter().filter_map(move |&(matrix, layer)| {
            let out = self.shapes.shapes[matrix.index()].out;
            (out > 0).then(|| {
                let base = channel_base(matrix, layer, log2_l_pad, log2_c);
                let first = (base >> log2_k) as u64;
                let last = ((base + u128::from(out) - 1) >> log2_k) as u64;
                (first, last)
            })
        });
        core::iter::from_fn(move || {
            for (first, last) in ranges.by_ref() {
                match merged {
                    Some((low, high)) if first <= high => merged = Some((low, high.max(last))),
                    Some(done) => {
                        merged = Some((first, last));
                        return Some(done);
                    }
                    None => merged = Some((first, last)),
                }
            }
            merged.take()
        })
    }

    /// Whether `D` has a weight of type `matrix` at `layer`.
    fn is_weight(&self, matrix: WeightMatrix, layer: u32) -> bool {
        has_weight(&self.shapes, matrix, layer)
    }

    /// The canonical order of the weights: `Embed`, then `Q` to `Down` layer by layer, then
    /// `LmHead`.
    fn weights_in_order(&self) -> impl Iterator<Item = (WeightMatrix, u32)> + '_ {
        WeightMatrix::WEIGHTS.into_iter().flat_map(move |matrix| {
            let layers = if matrix.per_layer() {
                self.shapes.layers
            } else {
                1
            };
            (0..layers).map(move |layer| (matrix, layer))
        })
    }

    /// The position of `matrix`'s first tensor in the canonical order.
    fn first_tensor(&self, matrix: WeightMatrix) -> usize {
        match matrix {
            WeightMatrix::Embed => 0,
            WeightMatrix::LmHead => 1 + 7 * self.shapes.layers as usize,
            _ => 1 + (matrix.index() - 1) * self.shapes.layers as usize,
        }
    }
}

/// `log2` of the next power of two of `n`: 0 for `n ≤ 1`.
fn log2_pad(n: u32) -> u32 {
    u64::from(n).next_power_of_two().trailing_zeros()
}

/// The code of a matrix of `2^log2_k` values per row: [`PcsError::InverseRate`] for rate 1,
/// then [`PcsError::TooLong`] for codewords of more than `2^32` symbols.
fn check_code(params: &WeightLayoutParams, log2_k: u32) -> Result<(), PcsError> {
    if params.log2_inv_rate == 0 {
        return Err(PcsError::InverseRate { log2_inv_rate: 0 });
    }
    let log_len = log2_k + u32::from(params.log2_inv_rate);
    if log_len > Goldilocks::TWO_ADICITY {
        return Err(PcsError::TooLong { log_len });
    }
    Ok(())
}

/// Whether the shapes have a weight of type `matrix` at `layer`: a weight type, at layer 0 for
/// `Embed` and `LmHead`, below `n_layers` for the others.
fn has_weight(shapes: &WeightShapes, matrix: WeightMatrix, layer: u32) -> bool {
    match matrix {
        WeightMatrix::ChannelParams => false,
        _ if matrix.per_layer() => layer < shapes.layers,
        _ => layer == 0,
    }
}

/// `(t·L_pad + ℓ)·C`, the flattened index of channel 0 of weight `(t, ℓ)`: below `2^68`.
fn channel_base(matrix: WeightMatrix, layer: u32, log2_l_pad: u32, log2_c: u32) -> u128 {
    (((matrix.index() as u128) << log2_l_pad) + u128::from(layer)) << log2_c
}

/// The weights and channel parameters of `D` (03 §2), borrowed for the time of a commitment.
///
/// They are secret-class (05 §3): the `Debug` form shows the matrices, layers and counts, never
/// an entry.
#[derive(Clone, Default)]
pub struct ModelWeights<'a> {
    /// Every weight tensor: one per type and layer, in any order.
    pub tensors: Vec<WeightTensorRef<'a>>,
    /// Every `ChannelParams`: one for each weight the shapes list, in any order.
    pub channels: Vec<ChannelParamsRef<'a>>,
}

/// Shows the counts.
impl fmt::Debug for ModelWeights<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelWeights")
            .field("tensors", &self.tensors)
            .field("channels", &self.channels)
            .finish()
    }
}

/// One weight of `D`: its type, its layer (0 for `Embed` and `LmHead`), and its `out × in`
/// INT8 entries, row-major, rows being output channels (03 §2).
#[derive(Clone, Copy)]
pub struct WeightTensorRef<'a> {
    /// The weight type.
    pub matrix: WeightMatrix,
    /// The layer.
    pub layer: u32,
    /// The entries, row-major.
    pub data: &'a [i8],
}

/// Shows the weight and its length, never an entry.
impl fmt::Debug for WeightTensorRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightTensorRef")
            .field("matrix", &self.matrix)
            .field("layer", &self.layer)
            .field("len", &self.data.len())
            .finish()
    }
}

/// One weight's channel parameters (03 §2 `ChannelParams`): one value per output channel.
#[derive(Clone, Copy)]
pub struct ChannelParamsRef<'a> {
    /// The weight type.
    pub matrix: WeightMatrix,
    /// The weight's layer.
    pub layer: u32,
    /// The values.
    pub values: ChannelValues<'a>,
}

/// Shows the weight and the kind and count of its values, never a value.
impl fmt::Debug for ChannelParamsRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChannelParamsRef")
            .field("matrix", &self.matrix)
            .field("layer", &self.layer)
            .field("values", &self.values)
            .finish()
    }
}

/// A weight's channel parameters: class indices at a site of several classes, multipliers at an
/// affine site (03 §2, RFC-0025).
#[derive(Clone, Copy)]
pub enum ChannelValues<'a> {
    /// Each output channel's index into its site's shift classes.
    Classes(&'a [u8]),
    /// Each output channel's multiplier `M_c`.
    Multipliers(&'a [u16]),
}

impl ChannelValues<'_> {
    /// The number of values: one per output channel.
    pub fn len(&self) -> usize {
        match self {
            Self::Classes(values) => values.len(),
            Self::Multipliers(values) => values.len(),
        }
    }

    /// Whether there is no value.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Value `c`, in `Fp`.
    fn get(&self, c: usize) -> Fp {
        match self {
            Self::Classes(values) => Fp::from_u64_reduced(u64::from(values[c])),
            Self::Multipliers(values) => Fp::from_u64_reduced(u64::from(values[c])),
        }
    }
}

/// Shows the kind and count.
impl fmt::Debug for ChannelValues<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self {
            Self::Classes(_) => "Classes",
            Self::Multipliers(_) => "Multipliers",
        };
        f.debug_struct(kind).field("len", &self.len()).finish()
    }
}

/// `D`'s weights, checked against the shapes: the tensors in canonical order and the channel
/// parameters in the layout's order.
pub(crate) struct Checked<'a> {
    tensors: Vec<&'a [i8]>,
    channels: Vec<ChannelValues<'a>>,
}

/// The committed rows of one matrix of `D`, laid out from its weights one row at a time, as the
/// fold pass reads them ([`WeightLayout::rows`]): row `j`, the `j`-th committed row in
/// increasing order, holds the entries RFC-0040 'The matrices' places in it and 0 at every
/// padding position, as the commitment's rows do.
///
/// It borrows the weights, secret-class values (05 §3), and keeps no copy: each row is written
/// into the buffer of the call. Its `Debug` form shows the matrix and the rows' count.
pub struct WeightRows<'l, 'w> {
    layout: &'l WeightLayout,
    matrix: WeightMatrix,
    checked: Checked<'w>,
    /// The channel parameters' committed rows, whose numbers follow no formula; empty for a
    /// weight type, whose row `j` is `(⌊j / ⌈out / 2^a⌉⌋, j mod ⌈out / 2^a⌉)`.
    rows: Vec<u64>,
}

/// Shows the matrix and its committed rows' count.
impl fmt::Debug for WeightRows<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightRows")
            .field("matrix", &self.matrix)
            .field("rows", &self.layout.matrix(self.matrix).rows)
            .finish_non_exhaustive()
    }
}

impl MatrixRows for WeightRows<'_, '_> {
    fn matrix(&self) -> WeightMatrix {
        self.matrix
    }

    /// Writes committed row `index` over `out`, which holds `k_T` values.
    ///
    /// # Errors
    ///
    /// In this order: [`PcsError::RowIndex`] for an index of `m_T` or more, and
    /// [`PcsError::RowLength`] unless `out` holds `k_T` values. Nothing is written on error.
    fn read_row(&mut self, index: usize, out: &mut [Fp]) -> Result<(), PcsError> {
        let layout = &self.layout.matrices[self.matrix.index()];
        if index as u64 >= layout.rows {
            return Err(PcsError::RowIndex {
                matrix: self.matrix,
                index,
                rows: layout.rows,
            });
        }
        if out.len() != layout.message_len() {
            return Err(PcsError::RowLength {
                expected: layout.message_len(),
                found: out.len(),
            });
        }
        let row = match layout.kind {
            MatrixKind::Weight { layers, h, .. } => {
                let per_layer = layout.rows / u64::from(layers);
                let index = index as u64;
                ((index / per_layer) << h) + index % per_layer
            }
            MatrixKind::ChannelParams { .. } => self.rows[index],
        };
        self.layout
            .fill_row(self.matrix, &self.checked, row, index, out);
        Ok(())
    }
}
