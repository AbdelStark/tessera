// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The degree bound of a sumcheck, a function of its protocol and its kind (RFC-0013 'Masked
//! extensions' 'Degree', RFC-0014 'Soundness budget').

use crate::Prefix;

/// The protocol a sumcheck is proven in, which sets its degree bound with the sumcheck's kind
/// ([`Protocol::max_degree`]).
///
/// `tessera-protocol/2` reads every committed or virtual factor of a reduction stage through its
/// masked extension, which adds 2 to that factor's degree in the round that binds its last
/// variable, so its reduction stages reach degree 10 (RFC-0013 'Degree'). Everything else stays
/// at 7.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Protocol {
    /// `tessera-protocol/1`, sound and not hiding.
    One,
    /// `tessera-protocol/2`, with masked sumchecks and masked extensions (RFC-0013).
    Two,
}

/// The bound of every sumcheck but a reduction stage of `tessera-protocol/2`: a product of up to
/// seven multilinears, RFC-0014's `d ≤ 7`.
const BOUND: usize = 7;

/// The bound of a reduction stage of `tessera-protocol/2`: RFC-0013 'Degree''s largest row,
/// RFC-0050's root coefficient products and RFC-0066's `Xor3` and `Maj` with three masked reads.
const REDUCTION_BOUND_2: usize = 10;

// The round polynomials hold every degree a protocol allows.
const _: () = assert!(REDUCTION_BOUND_2 == tessera_poly::MAX_DEGREE && BOUND < REDUCTION_BOUND_2);

impl Protocol {
    /// The largest degree a sumcheck under `prefix` has in this protocol (RFC-0013 'Masked
    /// extensions' 'Degree' 'The invariant'):
    ///
    /// - 10 at a reduction stage of `tessera-protocol/2`, `stage/{k}` with `k ≥ 1` (RFC-0010
    ///   'Stages': reduction stages are indexed from 1);
    /// - 7 everywhere else: every sumcheck of `tessera-protocol/1`, and in `tessera-protocol/2`
    ///   the tree levels (RFC-0032), stage 0's layer sumcheck (RFC-0068), and the merge of a
    ///   grouped leaf stage (RFC-0054).
    ///
    /// [`prove_batch`](crate::prove_batch) refuses a degree above it with
    /// [`SumcheckError::Degree`](crate::SumcheckError::Degree).
    pub const fn max_degree(self, prefix: Prefix) -> usize {
        match (self, prefix) {
            (Self::Two, Prefix::Stage(k)) if k >= 1 => REDUCTION_BOUND_2,
            _ => BOUND,
        }
    }

    /// The `SessionContext.protocol` string that binds the protocol (RFC-0004 'Transcript').
    pub const fn name(self) -> &'static str {
        match self {
            Self::One => "tessera-protocol/1",
            Self::Two => "tessera-protocol/2",
        }
    }
}
