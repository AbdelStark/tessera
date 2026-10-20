// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The weight commitment opened by the folding pass (RFC-0012 'Opening by the folding pass',
//! RFC-0049, RFC-0052), as `tessera-protocol/1` opens it: one claim per weight matrix
//! `Embed`..`LmHead`, `m ≥ 1` claims in pool order on `ChannelParams`, no mask and no blinding row.
//!
//! # The claim and its point
//!
//! A matrix `T`'s claim is `Ã_T(point) = v`, `Ã_T` the extension of its full matrix, the
//! committed rows and the structurally zero ones, flattened row by row (RFC-0010 'Multilinear
//! extensions'). The point's low `log2 k_T` coordinates are the column point, the others the row
//! point. A weight type's matrix flattens to its weights `W[ℓ, o, i]` padded to `L_pad × out_pad ×
//! in_pad`: row `ℓ·2^{h_T} + ⌊o / 2^{a_T}⌋` and column `(o mod 2^{a_T})·in_pad + i` make the
//! index `(ℓ·out_pad + o)·in_pad + i` (RFC-0040 'The matrices'). So its claim `W̃_T(r_i ‖ r_o ‖
//! ρ_ℓ)` from the `MatMulWeight` stage (RFC-0010, RFC-0037 'The claim pool') has the column point
//! `r_i ‖ r_o^lo` and the row point `r_o^hi ‖ ρ_ℓ`, `r_o^lo` the low `a_T` coordinates of `r_o`.
//!
//! # The pass
//!
//! [`WeightFold::new`] takes the point's coordinates that are known when the pass starts, the
//! fold point: `r_o` for a weight type, when its `MatMulWeight` instance starts, and the whole
//! row point for `ChannelParams`, whose claim has no later coordinate. It takes `ξ*_T` too,
//! squeezed right after the seed (RFC-0049 'The schedule'). It reads each committed row once, in
//! order, through a [`MatrixRows`], and accumulates two things from it:
//!
//! - **The folds** `F_ℓ(o_lo, i) = Σ_{o_hi} eq(r_o^hi, o_hi)·W_ℓ[(o_hi, o_lo), i]`, one row of
//!   `k_T` values per layer (RFC-0012 step 1). `ChannelParams` has one, `Σ_r eq(row point, r)·A_r`.
//! - **The proximity row** `u_ξ,T = Σ_{j < m_T} eq(ξ*_T, j)·W_T[row_j]`, `row_j` the `j`-th
//!   committed row and `ξ*_T` of `⌈log2 m_T⌉` coordinates (RFC-0049 'The proximity row').
//!
//! No later step reads a weight (INV-WC-4):
//!
//! - [`WeightFold::sumcheck_input`] is the matmul sumcheck's `W̃(ℓ, r_o, i) = Σ_{o_lo}
//!   eq(r_o^lo, o_lo)·F_ℓ(o_lo, i)`, `2^a·in_pad` products per layer (RFC-0012 step 2). For
//!   `ChannelParams` it is the fold itself, the matrix's columns at the row point.
//! - [`WeightFold::combined_rows`] gives, once `ρ_ℓ` and `θ_T` are drawn, the evaluation row
//!   `u_T = θ_T·Σ_ℓ eq(ρ_ℓ, ℓ)·F_ℓ` (RFC-0012 step 3, with no blinding row) and the proximity
//!   row, neither with a mask: [`WeightCombinedRows`].
//!
//! The column answers are the commitment's ([`WeightCommitment::matrix`] and
//! [`ReferenceCommitment::answer`]): `tessera-protocol/1` holds the codewords in memory, and
//! reads no weight to answer (RFC-0012 'Column answers').
//!
//! # The verifier's checks
//!
//! As RFC-0011 steps 6 and 8 check the advice's, with RFC-0049's coefficients: `θ_T·v =
//! ⟨u_T, eq(column point)⟩`; then, at each queried column `j`, `Σ_r θ_T·eq(row point, r)·col_j[r]
//! = enc(u_T)(x_j)` over the committed rows `r`, and `Σ_j' eq(ξ*_T, j')·col_j[j'] =
//! enc(u_ξ,T)(x_j)` by position. A matrix with no committed row has no claim, no combined rows
//! and no `ξ*_T` (RFC-0049).
//!
//! # Secrets
//!
//! The weights, the rows the pass reads, the folds and the proximity row until it is sent are
//! secret-class (05 §3): the pass reads each row into a [`SecretBuf`] and keeps the folds and the
//! proximity row in `SecretBuf`s, all cleared when they drop (OBS-6). Combined rows are
//! protocol-class once returned: they are the message. The points, `θ_T` and the coefficients are
//! public, from the transcript. Control flow depends on the layout and the points' lengths, never
//! on a value (INV-ARCH-2), and the `Debug` forms show shapes only.
//!
//! [`WeightCommitment::matrix`]: crate::WeightCommitment::matrix
//! [`ReferenceCommitment::answer`]: crate::ReferenceCommitment::answer

use core::fmt;

use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, QuadraticExtension};

use crate::opening::add_scaled_eq;
use crate::{CombinedRow, MatrixKind, PcsError, WeightLayout, WeightMatrix};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

/// The committed rows of one weight matrix, as the fold pass reads them: each by its position
/// `j` among the matrix's `m_T` committed rows, in RFC-0040's increasing order, written into a
/// buffer of `k_T` values.
///
/// [`WeightLayout::rows`] lays them out from `D`'s weights ([`WeightRows`](crate::WeightRows)).
/// The rows are secret-class (05 §3): an implementation writes a row only into the buffer it is
/// given, which [`WeightFold::new`] keeps in a [`SecretBuf`].
pub trait MatrixRows {
    /// The matrix whose rows these are.
    fn matrix(&self) -> WeightMatrix;

    /// Writes committed row `index` over `out`, which holds `k_T` values: every entry, 0 at a
    /// padding position.
    ///
    /// # Errors
    ///
    /// An implementation refuses a row past the matrix's committed rows and a buffer that does
    /// not hold `k_T` values; [`WeightRows`](crate::WeightRows)' errors are
    /// [`PcsError::RowIndex`] and [`PcsError::RowLength`].
    fn read_row(&mut self, index: usize, out: &mut [Fp]) -> Result<(), PcsError>;
}

/// The rows a reference lends.
impl<R: MatrixRows + ?Sized> MatrixRows for &mut R {
    fn matrix(&self) -> WeightMatrix {
        (**self).matrix()
    }

    fn read_row(&mut self, index: usize, out: &mut [Fp]) -> Result<(), PcsError> {
        (**self).read_row(index, out)
    }
}

/// A weight matrix's combined rows, RFC-0036's and RFC-0052's `WeightCombinedRows`: the
/// evaluation rows (`1` for `Embed`..`LmHead`, `m ≥ 1` in pool order for `ChannelParams`) and the
/// proximity row `u_ξ,T`, `k_T` values each, with no mask in `tessera-protocol/1`.
///
/// Protocol-class (05 §3): the `Debug` form shows the matrix and the rows' lengths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightCombinedRows {
    /// The matrix.
    pub matrix: WeightMatrix,
    /// The evaluation rows: `u_T = θ_T·Σ_ℓ eq(ρ_ℓ, ℓ)·F_ℓ` for a weight type, and
    /// `u_{CP, i} = θ_CP·F_i` for each claim `i ∈ 0..m` on `ChannelParams` (RFC-0052).
    pub evaluations: Vec<CombinedRow>,
    /// `u_ξ,T = Σ_{j < m_T} eq(ξ*_T, j)·W_T[row_j]`: the committed rows combined by position.
    pub proximity: CombinedRow,
}

/// One pass over a weight matrix's committed rows (RFC-0012 'Opening by the folding pass',
/// RFC-0049): the per-layer folds `F_ℓ` and the proximity row, from which the sumcheck input and
/// both combined rows follow without another read of the weights. The [crate
/// documentation](crate#the-weight-opening) states each.
///
/// The folds and the proximity row are secret-class until sent (05 §3): they live in
/// `SecretBuf`s, cleared when the fold drops; there is no `Clone`, and the `Debug` form shows the
/// shape.
pub struct WeightFold {
    matrix: WeightMatrix,
    /// `k_T`.
    k: usize,
    /// The column bits the fold point fixes: `a_T`, 0 for `ChannelParams`.
    bound: u32,
    /// The layers with committed rows: `n_layers` for `Q` to `Down`, 1 for the others.
    layers: usize,
    /// `log2 L_pad`: the row point's layer coordinates, which the evaluation row takes.
    layer_vars: u32,
    /// The fold point's column coordinates, `r_o^lo`.
    columns: Vec<Fp2>,
    /// `F_ℓ` for each layer: value `c` of layer `ℓ` at `ℓ·k_T + c`.
    folds: SecretBuf<Fp2>,
    /// `u_ξ,T`, `k_T` values.
    proximity: SecretBuf<Fp2>,
}

/// Shows the shape: the folds and the proximity row are secret until sent.
impl fmt::Debug for WeightFold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightFold")
            .field("matrix", &self.matrix)
            .field("message_len", &self.k)
            .field("layers", &self.layers)
            .finish_non_exhaustive()
    }
}

impl WeightFold {
    /// The fold pass of `matrix` on `layout` (RFC-0012 step 1, RFC-0049): reads each of its
    /// `m_T` committed rows once from `rows`, in order, into the folds `F_ℓ` and the proximity
    /// row.
    ///
    /// `point` is the fold point, the claim's coordinates known when the pass starts, lowest
    /// first: for a weight type `r_o`, its `log2 out_pad = a_T + h_T` output coordinates
    /// (`r_o^lo` the first `a_T`); for `ChannelParams` the claim's row point, the row index's
    /// bits. `xi_point` is `ξ*_T`, `⌈log2 m_T⌉` coordinates.
    ///
    /// It allocates `(L + 1)·k_T` values of `Fp2` for `L` layers, and `k_T` of `Fp` for the row
    /// it reads into, and computes `2·m_T·k_T` products of an `Fp2` coefficient by an `Fp` value.
    ///
    /// # Errors
    ///
    /// In this order, before any row is read: [`PcsError::NoRows`] for a matrix with no
    /// committed row, which has no combined rows and no `ξ*_T`; [`PcsError::FoldPoint`] unless
    /// `point` has the fold point's coordinates; [`PcsError::XiPoint`] unless `xi_point` has
    /// `⌈log2 m_T⌉`; [`PcsError::OtherMatrix`] for rows of another matrix; and
    /// [`PcsError::TooLarge`] for folds no buffer can hold. Then the first error `rows` gives.
    pub fn new<R: MatrixRows + ?Sized>(
        layout: &WeightLayout,
        matrix: WeightMatrix,
        rows: &mut R,
        point: &[Fp2],
        xi_point: &[Fp2],
    ) -> Result<Self, PcsError> {
        if matrix == WeightMatrix::ChannelParams {
            return Self::channel_params(layout, rows, &[point], xi_point);
        }
        let shape = layout.matrix(matrix);
        let count = shape.rows();
        if count == 0 {
            return Err(PcsError::NoRows { matrix });
        }
        let MatrixKind::Weight { a, layers, h, .. } = shape.kind() else {
            unreachable!();
        };
        let (bound, layers, fold_vars) = (a, layers as usize, h);
        let layer_vars = shape.row_vars() - fold_vars;
        let expected = (bound + fold_vars) as usize;
        if point.len() != expected {
            return Err(PcsError::FoldPoint {
                matrix,
                expected,
                found: point.len(),
            });
        }
        // ⌈log2 m_T⌉, m_T ≥ 1: the bits of a position below m_T.
        let xi_vars = (u64::BITS - (count - 1).leading_zeros()) as usize;
        if xi_point.len() != xi_vars {
            return Err(PcsError::XiPoint {
                expected: xi_vars,
                found: xi_point.len(),
            });
        }
        if rows.matrix() != matrix {
            return Err(PcsError::OtherMatrix {
                expected: matrix,
                found: rows.matrix(),
            });
        }
        let k = shape.message_len();
        let too_large = PcsError::TooLarge {
            rows: count,
            row_len: k,
        };
        let m = usize::try_from(count).map_err(|_| too_large)?;
        let size = layers.checked_mul(k).ok_or(too_large)?;

        // The coefficients, public: eq(row point, f_j) at each committed row's bits below the
        // layer index, f_j, and eq(ξ*_T, j) at its position.
        let (columns, row_point) = point.split_at(bound as usize);
        let committed: Vec<u64> = layout.committed_rows(matrix).collect();
        let low_bits = |row: u64| {
            row.checked_shr(fold_vars)
                .map_or(row, |high| row - (high << fold_vars))
        };
        let mut fold_weights = vec![Fp2::ZERO; m];
        add_scaled_eq(
            &mut fold_weights,
            committed.iter().map(|&row| low_bits(row)),
            row_point,
            Fp2::ONE,
        );
        let mut xi_weights = vec![Fp2::ZERO; m];
        add_scaled_eq(&mut xi_weights, 0..count, xi_point, Fp2::ONE);

        // The pass: each committed row once, into its layer's fold and the proximity row.
        let mut folds = SecretBuf::filled(Fp2::ZERO, size);
        let mut proximity = SecretBuf::filled(Fp2::ZERO, k);
        let mut row = SecretBuf::filled(Fp::ZERO, k);
        for (j, (&number, (&w_fold, &w_xi))) in committed
            .iter()
            .zip(fold_weights.iter().zip(&xi_weights))
            .enumerate()
        {
            rows.read_row(j, &mut row)?;
            // The layer index, ℓ < L: the row's bits from fold_vars up.
            let layer = number.checked_shr(fold_vars).unwrap_or(0) as usize;
            let fold = &mut folds[layer * k..][..k];
            for ((f, p), &x) in fold.iter_mut().zip(proximity.iter_mut()).zip(row.iter()) {
                *f += w_fold.mul_base(x);
                *p += w_xi.mul_base(x);
            }
        }
        Ok(Self {
            matrix,
            k,
            bound,
            layers,
            layer_vars,
            columns: columns.to_vec(),
            folds,
            proximity,
        })
    }

    /// The single-pass fold of `ChannelParams` across `m ≥ 1` claims in pool order (RFC-0052
    /// 'Opening by the single-pass fold'): reads each committed row of `ChannelParams` once from
    /// `rows`, in order, accumulating `F_i = Σ_{j < m_CP} eq(row_points[i], r_j)·W_CP[row_j]` for
    /// every `i ∈ 0..m` and the proximity row `u_{ξ, CP}`.
    ///
    /// # Errors
    ///
    /// In this order, before any row is read: [`PcsError::NoRows`] if `ChannelParams` has no
    /// committed row, [`PcsError::NoClaims`] if `row_points` is empty, [`PcsError::FoldPoint`]
    /// unless every point in `row_points` has `row_vars` coordinates, [`PcsError::XiPoint`] unless
    /// `xi_point` has `⌈log2 m_CP⌉` coordinates, [`PcsError::OtherMatrix`] if `rows` is not for
    /// `ChannelParams`, and [`PcsError::TooLarge`] if the fold buffers do not fit in memory. Then
    /// the first error `rows` gives.
    pub fn channel_params<R: MatrixRows + ?Sized>(
        layout: &WeightLayout,
        rows: &mut R,
        row_points: &[&[Fp2]],
        xi_point: &[Fp2],
    ) -> Result<Self, PcsError> {
        let matrix = WeightMatrix::ChannelParams;
        let shape = layout.matrix(matrix);
        let count = shape.rows();
        if count == 0 {
            return Err(PcsError::NoRows { matrix });
        }
        if row_points.is_empty() {
            return Err(PcsError::NoClaims);
        }
        let expected = shape.row_vars() as usize;
        for &point in row_points {
            if point.len() != expected {
                return Err(PcsError::FoldPoint {
                    matrix,
                    expected,
                    found: point.len(),
                });
            }
        }
        let xi_vars = (u64::BITS - (count - 1).leading_zeros()) as usize;
        if xi_point.len() != xi_vars {
            return Err(PcsError::XiPoint {
                expected: xi_vars,
                found: xi_point.len(),
            });
        }
        if rows.matrix() != matrix {
            return Err(PcsError::OtherMatrix {
                expected: matrix,
                found: rows.matrix(),
            });
        }
        let k = shape.message_len();
        let too_large = PcsError::TooLarge {
            rows: count,
            row_len: k,
        };
        let m = usize::try_from(count).map_err(|_| too_large)?;
        let claims = row_points.len();
        let size = claims.checked_mul(k).ok_or(too_large)?;
        let weights_len = claims.checked_mul(m).ok_or(too_large)?;

        let committed: Vec<u64> = layout.committed_rows(matrix).collect();
        let mut fold_weights = vec![Fp2::ZERO; weights_len];
        for (weights, &row_point) in fold_weights.chunks_exact_mut(m).zip(row_points) {
            add_scaled_eq(weights, committed.iter().copied(), row_point, Fp2::ONE);
        }
        let mut xi_weights = vec![Fp2::ZERO; m];
        add_scaled_eq(&mut xi_weights, 0..count, xi_point, Fp2::ONE);

        let mut folds = SecretBuf::filled(Fp2::ZERO, size);
        let mut proximity = SecretBuf::filled(Fp2::ZERO, k);
        let mut row = SecretBuf::filled(Fp::ZERO, k);
        for (j, &w_xi) in xi_weights.iter().enumerate() {
            rows.read_row(j, &mut row)?;
            for (p, &x) in proximity.iter_mut().zip(row.iter()) {
                *p += w_xi.mul_base(x);
            }
            for (fold, weights) in folds.chunks_exact_mut(k).zip(fold_weights.chunks_exact(m)) {
                let w_fold = weights[j];
                for (f, &x) in fold.iter_mut().zip(row.iter()) {
                    *f += w_fold.mul_base(x);
                }
            }
        }
        Ok(Self {
            matrix,
            k,
            bound: 0,
            layers: claims,
            layer_vars: 0,
            columns: Vec::new(),
            folds,
            proximity,
        })
    }

    /// The matrix.
    pub fn matrix(&self) -> WeightMatrix {
        self.matrix
    }

    /// The table the fold point leaves, the matmul sumcheck's input (RFC-0012 step 2):
    /// `W̃(ℓ, r_o, i) = Σ_{o_lo} eq(r_o^lo, o_lo)·F_ℓ(o_lo, i)` at `ℓ·in_pad + i`, `2^{n_ℓ + n_i}`
    /// values, the layer index high and padded layers 0. For `ChannelParams`, the fold itself:
    /// the matrix's `k_T` columns at the row point, `Ã(c ‖ row point)`.
    ///
    /// It reads no weight: `2^{a_T}·in_pad` products per layer, from the folds. The table is
    /// secret-class, in a [`SecretBuf`].
    pub fn sumcheck_input(&self) -> SecretBuf<Fp2> {
        if self.matrix == WeightMatrix::ChannelParams {
            let mut table = SecretBuf::filled(Fp2::ZERO, self.folds.len());
            table.copy_from_slice(&self.folds);
            return table;
        }
        let free = self.k >> self.bound;
        let mut eq_columns = vec![Fp2::ZERO; 1 << self.bound];
        add_scaled_eq(
            &mut eq_columns,
            0..1u64 << self.bound,
            &self.columns,
            Fp2::ONE,
        );
        let mut table = SecretBuf::filled(Fp2::ZERO, free << self.layer_vars);
        for (fold, out) in self
            .folds
            .chunks_exact(self.k)
            .zip(table.chunks_exact_mut(free))
        {
            for (block, &e) in fold.chunks_exact(free).zip(&eq_columns) {
                for (t, &f) in out.iter_mut().zip(block) {
                    *t += e * f;
                }
            }
        }
        table
    }

    /// The combined rows, once the layer point `ρ_ℓ` and `θ_T` are drawn: the evaluation rows
    /// (`u_T = θ_T·Σ_ℓ eq(ρ_ℓ, ℓ)·F_ℓ` for a weight type, and `u_{CP, i} = θ_CP·F_i` for each claim
    /// on `ChannelParams`, RFC-0012 step 3 and RFC-0052) and the proximity row `u_ξ,T` (RFC-0049),
    /// with no mask in `tessera-protocol/1`. `layer_point` has `log2 L_pad` coordinates, the row
    /// point's above the fold point: none for `Embed`, `LmHead` and `ChannelParams`.
    ///
    /// It reads no weight: `L·k_T` products from the folds, and a copy of the proximity row.
    ///
    /// # Errors
    ///
    /// [`PcsError::LayerPoint`] unless `layer_point` has `log2 L_pad` coordinates.
    pub fn combined_rows(
        &self,
        layer_point: &[Fp2],
        theta: Fp2,
    ) -> Result<WeightCombinedRows, PcsError> {
        if layer_point.len() != self.layer_vars as usize {
            return Err(PcsError::LayerPoint {
                matrix: self.matrix,
                expected: self.layer_vars as usize,
                found: layer_point.len(),
            });
        }
        let evaluations = if self.matrix == WeightMatrix::ChannelParams {
            self.folds
                .chunks_exact(self.k)
                .map(|fold| CombinedRow {
                    u: fold.iter().map(|&f| theta * f).collect(),
                    mask: Vec::new(),
                })
                .collect()
        } else {
            let mut weights = vec![Fp2::ZERO; self.layers];
            add_scaled_eq(&mut weights, 0..self.layers as u64, layer_point, theta);
            let mut u = vec![Fp2::ZERO; self.k];
            for (fold, &w) in self.folds.chunks_exact(self.k).zip(&weights) {
                for (value, &f) in u.iter_mut().zip(fold) {
                    *value += w * f;
                }
            }
            vec![CombinedRow {
                u,
                mask: Vec::new(),
            }]
        };
        Ok(WeightCombinedRows {
            matrix: self.matrix,
            evaluations,
            proximity: CombinedRow {
                u: self.proximity.to_vec(),
                mask: Vec::new(),
            },
        })
    }
}
