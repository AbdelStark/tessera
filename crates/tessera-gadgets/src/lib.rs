// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Witness generation and constraint descriptions for the uniqueness gadgets and the anchor
//! circuits (RFC-0009, RFC-0015, RFC-0017).
//!
//! A gadget pins a committed value to the unique result of a non-arithmetic operation, by lookups
//! whose keys and values are affine in its input, its hidden class column, its committed tensors
//! and constants (RFC-0009). This version holds the requantization, range and choice gadgets:
//!
//! - [`Requant`]: `Requant1` and `RequantSplit` (RFC-0009 Lemmas 1 and 2, the split point of
//!   RFC-0028) and `RequantWide` (RFC-0024 Lemma 11), built from the compiler's
//!   [`Gadget`](tessera_compiler::Gadget) and the tables it names;
//! - [`NonNeg`]: `v ∈ [0, 2^k)` by limbs (RFC-0009 Lemma 3);
//! - [`BigIntCompare`]: carry normalization and exact comparison of limb vectors that may exceed
//!   the field as whole integers (RFC-0009 Lemma 4), with [`CompareConstraints`] for its zero-checks
//!   and range lookups, and a zeroizing [`CompareWitness`].
//! - [`SoftmaxRow`]: the causal maximum and reciprocal bracket, including masked exponentials
//!   and padded rows (RFC-0009 Lemmas 5–6, RFC-0030, RFC-0035). Its
//!   [`SoftmaxConstraints`] keep the nonlinear pointwise reductions explicit. Each row takes
//!   a public valid-prefix length, derived by the driver from the causal selector. Rows of one
//!   position check their single product leaf directly; taller rows use the product forest.
//! - [`RsqrtRow`]: the exact multi-limb root bracket (RFC-0009 Lemma 7), with zeroizing
//!   witnesses, bounded product coefficients and a padded-root zero-check.
//! - [`OneHot`] and [`Argmax`]: the token indicator and lowest-index greedy certificate
//!   (RFC-0009 Lemmas 8–9), with sumcheck relations and a public suffix kernel. Argmax
//!   borrows the next token's existing indicator (teacher forcing).
//! - [`VocabMax`] and [`InverseCdf`]: the vocabulary maximum/exponential certificate (RFC-0009
//!   Lemma 5 without reciprocal, RFC-0017, RFC-0030) and the temperature-sampling certificate
//!   (RFC-0009 Lemma 10) in RFC-0058's columns: three degree-3 vocabulary contractions, the
//!   radix-`2^16` limbs of the draw, the partition and the prefix, and two exact comparisons,
//!   with a branch-free honest sampler.
//!
//! The scalar gadgets (`Requant` and `NonNeg`) give:
//!
//! - **its constraints**, [`Constraints`]: the committed tensors, the virtual values and the
//!   lookups, each lookup's key and value an [`Affine`] form over [`Term`]s. The forms hold
//!   element by element, so the same form turns a LogUp claim on a key or value column at a point
//!   into claims on the input, the classes and the committed tensors there: what the leaf stage
//!   and the verifier's final checks do with them (RFC-0037).
//! - **its honest witness**, [`Witness`]: the committed tensors of the one assignment the lemma
//!   admits.
//!
//! [`Lookups`] then evaluates the constraints on a witness into the LogUp prover's columns, one
//! instance per table the gadget reads (RFC-0008 'Batching'), with the multiplicities an honest
//! prover commits and the tables' rows, [`TableColumns`].
//!
//! ```
//! use tessera_codec::Ascii;
//! use tessera_compiler::Gadget;
//! use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
//! use tessera_gadgets::{Committed, Lookups, Requant, TableColumns, Term};
//! use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
//! use tessera_logup::{TreeId, prove_logup};
//! use tessera_tables::{GeneratedTable, Table, TableId, TableKind};
//!
//! // Requant1 for keys of 4 signed bits, shifted by 2 and saturated to [−1, 1].
//! let kind = TableKind::Requant { b: 4, r_by_class: vec![2], lo: -1, hi: 1 };
//! let spec = GeneratedTable::new(TableId(0), kind)?.spec().clone();
//! let gadget = Requant::new(&Gadget::Requant1 { table: TableId(0) }, &[spec])?;
//! // Each key's output is ⌊(z + 2) / 4⌋, saturated.
//! let keys = [-8, -3, -2, 1, 2, 5, 6, 7];
//! let witness = gadget.witness(&keys, None)?;
//! assert_eq!(witness.tensor(Committed::Y), Some(&[-1, -1, 0, 0, 1, 1, 1, 1][..]));
//!
//! // The one lookup is (z + 2^3, y): one term on the input, the constant 8.
//! let constraints = gadget.constraints::<Goldilocks>();
//! let key = &constraints.lookups[0].key;
//! assert_eq!(key.constant, Goldilocks::from_u64_reduced(8));
//! assert_eq!(key.terms, [(Goldilocks::ONE, Term::Input)]);
//!
//! // Its LogUp instance, with the multiplicities an honest prover commits.
//! let lookups = Lookups::new(&constraints, &keys, None, &witness)?;
//! let tables = gadget
//!     .tables()
//!     .into_iter()
//!     .map(TableColumns::new)
//!     .collect::<Result<Vec<_>, _>>()?;
//! let multiplicities = lookups.multiplicities(&tables)?;
//! let instances = lookups.instances(&tables, &multiplicities)?;
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
//!     statement_digest: [0; 32],
//!     session_id: [0; 16],
//! };
//! let mut transcript = Transcript::new(&context);
//! let proven = prove_logup::<GoldilocksExt2, _>(
//!     &mut transcript,
//!     &instances,
//!     &[],
//!     &TreeId::paired(instances.len()),
//! )?;
//! assert_eq!(proven.claims().len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Secret-class data
//!
//! A gadget's input, its classes, its committed tensors and every column computed from them are
//! secret-class (05 §3). The input and the classes are lent to [`Requant::witness`],
//! [`NonNeg::witness`], [`BigIntCompare::witness`], [`RsqrtRow::witness`] and [`Lookups::new`] for the call. What they compute lives in
//! `SecretBuf`s, which clear their memory (OBS-6), and the debug forms show shapes only. Control
//! flow depends on public data only (INV-ARCH-2): an input outside the gadget's domain, the one
//! outcome that depends on the values, is reported after every element is computed. Errors carry
//! table identifiers, column names, lengths and rules, never a value or the position of the
//! element that failed.

#![forbid(unsafe_code)]

mod choice;
mod compare;
mod constraints;
mod error;
mod inverse_cdf;
mod lookups;
#[cfg(debug_assertions)]
pub mod malicious;
mod non_neg;
mod requant;
mod rsqrt;
mod softmax;
mod vocab;
mod witness;

pub use choice::{
    Argmax, ArgmaxEvaluation, ArgmaxRelations, ArgmaxWitness, OneHot, OneHotRelations,
    OneHotWitness, suffix_kernel,
};
pub use compare::{BigIntCompare, CompareColumn, CompareConstraints, CompareWitness, Side};
pub use constraints::{Affine, Committed, Constraints, Lookup, Term, Virtual};
pub use error::{Column, GadgetError};
pub use inverse_cdf::{
    BatchedCdfEvaluation, BatchedCdfSelector, CdfBatchColumns, CdfColumn, CdfConstraints,
    CdfEvaluation, CdfRelation, CdfRelations, CdfTerm, CdfWitness, Edge, InverseCdf,
    SampledRelations,
};
pub use lookups::{Group, Lookups, TableColumns};
pub use non_neg::NonNeg;
pub use requant::Requant;
pub use rsqrt::{
    Bracket, RsqrtColumn, RsqrtConstraints, RsqrtRelation, RsqrtRow, RsqrtTerm, RsqrtWitness,
};
pub use softmax::{
    Slack, SoftmaxColumn, SoftmaxConstraints, SoftmaxEvaluation, SoftmaxLookup, SoftmaxRelation,
    SoftmaxRow, SoftmaxTerm, SoftmaxValue, SoftmaxWitness,
};
pub use vocab::{
    VocabColumn, VocabConstraints, VocabEvaluation, VocabMax, VocabRelation, VocabRelations,
    VocabSelector, VocabTerm, VocabValue, VocabWitness,
};
pub use witness::Witness;
