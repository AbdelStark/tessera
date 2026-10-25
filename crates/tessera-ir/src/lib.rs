// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The typed integer dataflow IR and the claim-list types (RFC-0006, RFC-0007).
//!
//! A [`Graph`] holds tensors ([`TensorSpec`]), each with its shape, integer type, range and
//! [`NodeKind`], and nodes applying an [`Op`] of RFC-0006's closed operation set to them. A
//! [`GraphBuilder`] adds declarations and nodes in topological order and checks each against
//! the rules of RFC-0006 'Operation set': arity, shapes, the kind of every output, and the
//! public constants, shift classes and weights a node names. Ids are dense in creation order,
//! so two builds from the same calls are equal (INV-DM-5). [`Graph::validate`] checks the same
//! rules on a graph assembled otherwise. Errors are [`IrError`]s.
//!
//! The graph is the constraint system the proof schedule reads: an operation's [`OpClass`] says
//! whether its relation is linear, of degree 2 or gadget-backed, and [`Graph::producer`] walks
//! the virtual DAG between committed tensors (RFC-0010). Range analysis, gadget selection and
//! the Llama lowering build on it (RFC-0006 'Range analysis', 'Gadget selection', 'Llama
//! lowering').

#![forbid(unsafe_code)]

mod claim;
mod error;
mod files;
mod graph;
mod model;
mod op;
mod types;
mod weights;

pub use claim::{ClaimList, CommitmentLayout, GadgetPolicy, PROTOCOL_VERSION};
pub use error::IrError;
pub use files::{LogitsFile, TokenFile};
pub use graph::{Graph, GraphBuilder, Node};
pub use model::{
    ArchFamily, ArchSpec, CalibrationIdentity, ChannelParams, Digest, IntegerModel, Profile,
    Provenance, PublicQuantSpec, QuantPoint, QuantSpec, RopeSpec, Site, WeightTensor,
};
pub use op::{BoolFn, Broadcast, Op, OpClass};
pub use tessera_hash::ParamSetId;
// The claim list's commitment layouts are the commitment crate's, field for field (RFC-0040
// 'Migration / Rollout').
pub use tessera_pcs::{AdviceLayoutParams, WeightLayoutParams};
pub use tessera_tables::TableId;
pub use types::{
    CommitmentClass, ConstRef, GadgetId, IntRange, IntType, NodeId, NodeKind, PrfRef,
    ShiftClassRef, StageId, TensorDecl, TensorId, TensorSpec, WeightRef, WeightType,
};
