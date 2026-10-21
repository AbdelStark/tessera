// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent final relations of RFC-0009. A local relation check is not proof
//! acceptance: its returned obligations must be bound to the graph reductions,
//! canonical advice roles and prescribed range lookups before the opening checks.
//!
//! # Composition contract
//!
//! [`Plan`] derives graph gadget ids, tables, source ids and geometry from the
//! verifier's own lowering and selection (SP-7). It supplies [`requant`], [`onehot`],
//! [`root`] and [`softmax`] sites. [`argmax::Argmax`] receives its id from the
//! verifier-owned scoring certificate; [`Plan::argmax_outputs`] binds the selected
//! logit and the SAME next-token indicator to their original graph row coordinates.
//! Maximum claims remain explicit certificate-opening obligations.
//!
//! Check lookup key/value evaluations against the earlier LogUp leaf claims at
//! their prescribed points. A Softmax maximum-forest leaf has its own point and
//! public root `1 - psi`; it is not interchangeable with the lookup point. Feed
//! nonlinear relation instances to [`crate::sumcheck::verify_batch`] with the
//! prescribed degree. Its callback receives the instance's final point AFTER
//! front-padding has been removed. Preserve every returned graph, advice, weight,
//! lookup and certificate claim until the corresponding proof discharges it.
//!
//! Relations with an incoming evaluation use that claim's point; zero-checks use
//! the common stage zero-check point. Fix incoming claims before batching gamma.
//! For a layer template, combine fixed per-layer incoming claims with the layer
//! weights before gamma, and reduce the nonlinear relation over the layer axis.
//! Substituting late per-layer evaluations at a known layer point permits
//! cancellation and does not establish the stacked relation. The full verifier
//! driver owns this assembly and its soundness/resource accounting (#128).
//! [`stacked`] composes root and Softmax native targets over actual template
//! families, retaining original per-layer bindings and explicit stacked source debts.
//!
//! Root input simplification is conditional on BOTH residual padding axes being
//! zero; [`root::Root::residual_padding_instance`] retains that proof obligation.
//! In a stacked relation the real-layer selector also belongs in `psi`. Tensor
//! shape metadata and physical commitment row padding do not prove these zeros.
//!
//! # Errors
//!
//! Failed local affine checks and proof evaluation counts give
//! [`GadgetError::Rejected`] with the verifier-owned gadget id. Incorrect caller
//! geometry is [`GadgetError::Internal`]; failed reserves are
//! [`GadgetError::Memory`]. Nonlinear batched proof failures keep
//! [`crate::sumcheck::SumcheckError`] and its exact stage/round in the driver;
//! a failed mixed batch cannot identify an individual guilty gadget. Invalid
//! lookup membership fails earlier as LogUp, per RFC-0019 steps 7 then 8.

pub mod argmax;
pub mod argmax_outputs;
pub mod comparison;
pub mod decomposition;
pub mod onehot;
pub mod requant;
pub mod root;
pub mod root_products;
pub mod softmax;
pub mod stacked;

use crate::advice::Source;
use crate::claims::GadgetPolicy;
use crate::lowering::Graph;
use crate::selection::{self, Selection};
use crate::validation::ClaimsError;
use crate::validation::GadgetId;
use core::fmt;
use tv_field::Fp2;

/// A canonically identified source at its prescribed point.
#[derive(Clone, Copy)]
pub struct Input<'a> {
    /// A graph tensor or a node's canonical auxiliary. Virtual tensors need graph
    /// reduction; committed tensors and auxiliaries need openings.
    pub source: Source,
    /// Natural LSB-first coordinates, with any broadcast projection already applied.
    pub point: &'a [Fp2],
}

/// A source evaluation still owed to graph reduction or commitment opening.
#[derive(Clone, Copy)]
pub struct Claim<'a> {
    /// The source and its prescribed point.
    pub input: Input<'a>,
    /// Its claimed multilinear extension, not an integer at a general point.
    pub value: Fp2,
}

/// Gadget metadata owned by the verifier, never selected by proof messages.
/// The graph must be lowered at the proof's actual sequence length.
#[derive(Debug)]
pub struct Plan<'g> {
    graph: &'g Graph,
    selected: Selection,
}

impl<'g> Plan<'g> {
    /// Re-derive ranges, pattern choices and lookup tables from public metadata (SP-7).
    /// This does not validate a complete claim list or accept a proof.
    ///
    /// # Errors
    /// The same public-domain, allocation and internal errors as [`selection::derive`].
    pub fn derive(graph: &'g Graph, policy: GadgetPolicy) -> Result<Self, ClaimsError> {
        Ok(Self {
            graph,
            selected: selection::derive(graph, policy)?,
        })
    }
}

/// A failed gadget relation, or a failure of the verifier itself (04 §2/§3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GadgetError {
    /// `Reject(Gadget { gadget })`: the proof's evaluations do not satisfy the relation.
    Rejected {
        /// The verifier-derived gadget's index, never an identifier from the proof.
        gadget: GadgetId,
    },
    /// `Inconclusive(ERR-RES-MEMORY)`: public metadata could not be reserved.
    Memory,
    /// `Inconclusive(ERR-INTERNAL)`: inconsistent verifier-owned parameters or geometry.
    Internal,
}

impl GadgetError {
    /// The gadget payload of `REJ-GADGET`, or `None` for a verifier failure.
    pub const fn rejected_gadget(self) -> Option<GadgetId> {
        match self {
            Self::Rejected { gadget } => Some(gadget),
            Self::Memory | Self::Internal => None,
        }
    }

    /// The stable identifier of 04 §2/§3.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Rejected { .. } => "REJ-GADGET",
            Self::Memory => "ERR-RES-MEMORY",
            Self::Internal => "ERR-INTERNAL",
        }
    }
}

impl fmt::Display for GadgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Rejected { gadget } => write!(f, "gadget {} failed its final relation", gadget.0),
            Self::Memory => f.write_str("memory for gadget checks ran out"),
            Self::Internal => f.write_str("gadget checks failed internally"),
        }
    }
}
impl std::error::Error for GadgetError {}
