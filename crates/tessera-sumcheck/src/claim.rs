// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The crate's one claim type: a table's multilinear extension at a point (RFC-0010
//! 'Multilinear extensions').

use core::fmt;

/// A claim `T̃(point) = value` on a table `T` of `2^n` entries, `n = point.len()`: what a
/// sumcheck reduces its input claims to, and what the next stage or an opening checks (RFC-0010
/// 'Stages', 'Claim pool and openings').
///
/// The point lists its coordinates lowest variable first: coordinate `k` sets bit `k` of the
/// table's index (RFC-0010 'Multilinear extensions'). Which table a claim is on comes from where
/// it is found: a forest's level ([`ProvenLevel::next_claims`](crate::ProvenLevel::next_claims))
/// or leaves ([`ProvenForests::leaf_claims`](crate::ProvenForests::leaf_claims)), or a matmul's
/// operand, by the field that holds it ([`MatMulWeightClaims`](crate::MatMulWeightClaims),
/// [`MatMulActClaims`](crate::MatMulActClaims)).
///
/// The value is protocol-class data, a function of secret-class tables (05 §3): the debug form
/// shows the number of coordinates only.
#[derive(Clone, PartialEq, Eq)]
pub struct Claim<F> {
    /// The point, its lowest variable first.
    pub point: Vec<F>,
    /// The claimed value of the table's multilinear extension at the point.
    pub value: F,
}

/// Shows the number of coordinates, never a value (05 §3).
impl<F> fmt::Debug for Claim<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Claim")
            .field("vars", &self.point.len())
            .finish_non_exhaustive()
    }
}
