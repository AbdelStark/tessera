// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The LogUp-GKR lookup prover with multiplicities and tuple tables (RFC-0008).
//!
//! A [`LogUpInstance`] proves that each of its `N ≥ 1` lookups `a_0, …, a_{N−1}` is a row of its
//! public table `t_0, …, t_{T−1}`, each a pair compressed by the challenge `β` to
//! `key + β·value`, with committed multiplicities `m_j` (RFC-0008 'Relation'):
//!
//! ```text
//! Σ_{i<N} 1/(α − a_i) − Σ_j m_j/(α − t_j) = 0
//! ```
//!
//! [`prove_logup`] runs the LogUp part of the leaf stage on the transcript (RFC-0014 'Transcript
//! schedule', rows 4 and 5):
//!
//! 1. it squeezes `α` under `logup/alpha`, then `β` under `logup/beta`;
//! 2. it builds two fraction trees per instance, one per side, each padded with row 0
//!    (RFC-0008 'Fraction tree'): the lookup tree's `2^{h_a}` leaves `(1, α − a_i)`, then
//!    `(1, α − t_0)`, and the table tree's `2^{h_t}` leaves `(−m_j, α − t_j)`, then
//!    `(−m_j, α − t_0)`, `h_a = ⌈log2 N⌉` and `h_t = ⌈log2 T⌉`;
//! 3. it absorbs every instance's roots `n_a, d_a, n_t, d_t` under `logup/roots`;
//! 4. it proves the trees level by level, beside the stage's product forests, with
//!    [`tessera_sumcheck::prove_trees`], in the order a schedule of [`TreeId`]s gives: `λ`
//!    follows `γ` at each level where a fraction tree is present (RFC-0010 'Tree levels');
//! 5. it absorbs `v = valũe(ρ_a)` of each instance whose table has values under `logup/values`;
//! 6. it returns each instance's claims on its keys, values and multiplicities
//!    ([`InstanceClaims`]) and each forest's leaf claim, in a [`ProvenLogUp`].
//!
//! A verifier checks `d_a ≠ 0`, `d_t ≠ 0` and `n_a·d_t + n_t·d_a = 0`, the levels, then that the
//! lookup tree's leaf numerator is 1 and that the table tree's leaf denominator is
//! `α − t̃(ρ_t)` of the public table. A lookup outside the table makes the roots balance for at
//! most `N + T − 1` values of `α`, the compression by `β` included: the error is at most
//! `(N + T)/|Fp2|`.
//!
//! Multiplicities are committed before `α`, so they are an input. [`multiplicities`] counts what
//! an honest prover commits from each lookup's row, the lookup tree's padding counted on row 0.
//!
//! ```
//! use tessera_codec::Ascii;
//! use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField};
//! use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
//! use tessera_logup::{LogUpInstance, TreeId, Tuples, multiplicities, prove_logup};
//!
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
//!     statement_digest: [0; 32],
//!     session_id: [0; 16],
//! };
//! let fp = Goldilocks::from_u64_reduced;
//! // A table of 5 rows (key, key²), and 6 lookups of rows 4, 0, 2, 4, 3 and 1.
//! let keys: Vec<Goldilocks> = (0..5).map(fp).collect();
//! let squares: Vec<Goldilocks> = (0..5).map(|k| fp(k * k)).collect();
//! let table = Tuples::new(&keys, Some(&squares));
//! let rows = [4u32, 0, 2, 4, 3, 1];
//! let lookup_keys: Vec<Goldilocks> = rows.iter().map(|&r| keys[r as usize]).collect();
//! let lookup_values: Vec<Goldilocks> = rows.iter().map(|&r| squares[r as usize]).collect();
//! // 2^3 table leaves: row 0 also counts the lookup tree's 2 padding leaves.
//! let m = multiplicities(table, &rows)?;
//! assert_eq!(m, [3, 1, 1, 1, 2, 0, 0, 0].map(fp));
//! let instance =
//!     LogUpInstance::new(table, Tuples::new(&lookup_keys, Some(&lookup_values)), &m);
//!
//! let mut transcript = Transcript::new(&context);
//! let proven =
//!     prove_logup::<GoldilocksExt2, _>(&mut transcript, &[instance], &[], &TreeId::paired(1))?;
//! // Both trees have height 3: three levels, each with λ.
//! assert_eq!(proven.levels().len(), 3);
//! assert!(proven.levels().iter().all(|level| level.lambda().is_some()));
//! assert_eq!((proven.roots().len(), proven.values().len()), (4, 1));
//! let claims = &proven.claims()[0];
//! assert_eq!(claims.key.point.len(), 3);
//! assert_eq!(claims.multiplicity.point.len(), 3);
//! # Ok::<(), tessera_logup::LogUpError>(())
//! ```
//!
//! # Layout and cost
//!
//! The trees are built bottom-up in a field-aware layout: the lookup tree's leaf numerators, all
//! 1, are never stored, and the table tree's stay in the prime subfield; every denominator and
//! every numerator above the leaves is in `F`. A node costs about 3 products, the lookup tree's
//! level above its leaves 1. The levels' sumchecks bind each of a fraction tree's tables
//! `eq, N_0, N_1, D_0, D_1` once per round and multiply `eq` once per pair, 5 products per pair
//! and node. On the lookup tree's last level, whose numerators are 1, no term reads `N_0` or
//! `N_1`: 3 products per pair and node, and the two tables of 1 are evaluated once rather than
//! bound. The lookup tree's levels dominate: at `2^22` lookups into `2^16` rows, about 88 ns per
//! lookup on one Apple M4 Max core, 76 of them in the levels (`benches/throughput.rs`).
//!
//! Control flow depends on shapes only (INV-ARCH-2). Lookups, values, multiplicities and tree
//! nodes are secret-class values (05 §3): no debug form or error of this crate shows one, only
//! lengths and indices.
//!
//! The code is generic over [`tessera_poly::PolyField`], so it runs over Goldilocks and its
//! extension, and over `F65537` and its extension for the soundness-calibration tests (07 §4),
//! with any transcript that implements [`tessera_sumcheck::SumcheckTranscript`].

#![forbid(unsafe_code)]

mod error;
mod instance;
mod prover;

pub use error::{Column, LogUpError};
pub use instance::{LogUpInstance, Tuples, multiplicities};
pub use prover::{InstanceClaims, ProvenLogUp, TreeId, prove_logup};
