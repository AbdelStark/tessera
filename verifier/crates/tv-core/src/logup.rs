// SPDX-License-Identifier: Apache-2.0 OR MIT

//! LogUp-GKR's checks around the tree levels (RFC-0008 'Fraction tree', RFC-0019 'Check order'
//! step 7): the roots of every instance's two fraction trees before the first level, and the
//! leaf checks after the last. The crate documentation is their specification ('LogUp roots',
//! 'LogUp leaf checks').
//!
//! A LogUp instance gathers every lookup into one table definition (RFC-0008 'Batching'). Its
//! lookup tree holds a leaf `(1, α − a_i)` per lookup, and its table tree a leaf
//! `(−m_j, α − t_j)` per row; both join the levels of [`verify_level`](crate::tree::verify_level)
//! where their heights say, beside the product forests of the leaf stage. The caller, which holds
//! the schedule, runs the three steps in order on one transcript:
//!
//! 1. [`check_roots`]: `α`, `β`, the roots message, and the root claims the trees enter with;
//! 2. the levels, each tree joining at `H − h` with its root claims and leaving after level
//!    `H − 1` with its leaf claims;
//! 3. [`check_leaves`]: the values message, each instance's leaf checks against the table
//!    evaluation `t̃(ρ_t)`, which the caller computes, and the claims left for later stages.
//!
//! ```
//! use tv_core::logup::{Leaves, check_leaves, check_roots};
//! use tv_field::{Fp, Fp2};
//! use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
//!
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1")?,
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("test-small/1")?),
//!     statement_digest: [0; 32],
//!     session_id: [0; 16],
//! };
//! // One lookup into a table of one row, key 5 and no value: both trees are single leaves, so
//! // they join no level, and their roots are their leaves. The prover knows α only from the
//! // transcript, so it runs the same squeezes.
//! let row = Fp2::from(Fp::reduce(5));
//! let mut prover = Transcript::new(&context);
//! let alpha = prover.challenge_fp2(Ascii::new("logup/alpha")?)?;
//! // The lookup's leaf (1, α − 5) and the row's (−1, α − 5): its multiplicity is 1.
//! let roots = [Fp2::ONE, alpha - row, -Fp2::ONE, alpha - row];
//!
//! let mut transcript = Transcript::new(&context);
//! let checked = check_roots(&mut transcript, 1, &roots)?;
//! assert_eq!(checked.challenges.alpha, alpha);
//! let instance = &checked.instances[0];
//! let leaves = [Leaves {
//!     lookup: instance.lookup.clone(),
//!     table: instance.table.clone(),
//!     has_values: false,
//! }];
//! // t̃ of the one-row table, on no variable, is the row itself.
//! let claims = check_leaves(&mut transcript, &checked.challenges, &leaves, &[], |_, _| Some(row))?;
//! assert_eq!(claims.instances[0].key.value, row);
//! assert_eq!(claims.instances[0].multiplicity.value, Fp2::ONE);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::fmt;

use tv_field::Fp2;
use tv_hash::transcript::{NotAscii, Transcript, TranscriptError};

use crate::fiat_shamir::{self, Failure, FiatShamir};
use crate::tree::FractionClaim;

/// The label of `α` (RFC-0014 'Transcript schedule', row 4).
const ALPHA: &str = "logup/alpha";
/// The label of `β`.
const BETA: &str = "logup/beta";
/// The label of the roots message.
const ROOTS: &str = "logup/roots";
/// The label of the values message (RFC-0014 'Transcript schedule', row 5).
const VALUES: &str = "logup/values";

/// The values the roots message holds for each instance: `n_a, d_a, n_t, d_t`.
const ROOTS_PER_INSTANCE: usize = 4;

/// LogUp's challenges, shared by every instance (RFC-0008 'Batching').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Challenges {
    /// `α`, the point of the fractions' denominators `α − a`.
    pub alpha: Fp2,
    /// `β`, which compresses a pair to `key + β·value`.
    pub beta: Fp2,
}

/// The root claims of one instance's two trees, on no variable: the claims they enter the levels
/// with.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RootClaims {
    /// The lookup tree's root, `(n_a, d_a)`.
    pub lookup: FractionClaim,
    /// The table tree's root, `(n_t, d_t)`.
    pub table: FractionClaim,
}

/// The roots that passed every check of [`check_roots`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[must_use = "the roots are the prover's word until the tree levels and the leaf checks bind them to the leaves"]
pub struct Roots {
    /// `α` and `β`.
    pub challenges: Challenges,
    /// Each instance's root claims, in schedule order.
    pub instances: Vec<RootClaims>,
}

/// One instance's trees after their last level, as [`check_leaves`] takes them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Leaves {
    /// The lookup tree's leaf claims, `Ñ(ρ_a) = p_a` and `D̃(ρ_a) = q_a`.
    pub lookup: FractionClaim,
    /// The table tree's leaf claims, `Ñ(ρ_t) = p_t` and `D̃(ρ_t) = q_t`.
    pub table: FractionClaim,
    /// Whether the instance's table has values, so that the values message holds its `v`. A
    /// `Range` table has none.
    pub has_values: bool,
}

/// A claim `f̃(point) = value` on one of an instance's polynomials, over `point.len()` variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Claim {
    /// The point, its low variables first.
    pub point: Vec<Fp2>,
    /// The claimed value there.
    pub value: Fp2,
}

/// The claims one instance's leaf checks leave for later stages.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InstanceClaims {
    /// `kẽy(ρ_a) = α − q_a − β·v` on the lookup-side key polynomial, which is virtual and goes to
    /// the stage that produces the keys.
    pub key: Claim,
    /// `valũe(ρ_a) = v` on the value polynomial, committed advice; `None` for a table without
    /// values.
    pub value: Option<Claim>,
    /// `m̃(ρ_t) = −p_t` on the multiplicity polynomial, committed advice.
    pub multiplicity: Claim,
}

/// The leaf checks that passed every check of [`check_leaves`], and what they leave.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[must_use = "the leaf checks prove nothing until the key, value and multiplicity claims are checked by the stages and openings that take them"]
pub struct LeafClaims {
    /// Each instance's claims, in schedule order.
    pub instances: Vec<InstanceClaims>,
}

/// Why the LogUp roots or leaf checks rejected a proof or could not run: the first check that
/// failed, in the order of the crate documentation. Instances are counted from 0 in schedule
/// order.
///
/// [`is_rejection`](Self::is_rejection) says whose fault an error is. `RootsLength`,
/// `LookupDenominator`, `TableDenominator`, `Unbalanced`, `ValuesLength`, `LeafNumerator` and
/// `TableEvaluation` judge the proof: `Reject(Lookup { table })` in 04 §2, for the table of the
/// instance [`rejected_instance`](Self::rejected_instance) gives. `RootsWithoutInstance` and
/// `ValuesWithoutInstance` judge the proof too, but name no instance, so no table. The others
/// report verifier input faults (`Inconclusive(ERR-INTERNAL)`), except
/// `Memory` or `Transcript(TranscriptError::Memory)`, which is exhausted reader memory
/// (`Inconclusive(ERR-RES-MEMORY)`), never a rejection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LookupError {
    /// The roots message holds `found` values where the instances need `expected`, four each.
    RootsLength {
        /// The first instance whose four roots the message does not hold, or the last instance
        /// when it holds more.
        instance: usize,
        /// Four values per instance.
        expected: usize,
        /// The values the message holds.
        found: usize,
    },
    /// An instance's lookup-tree root has the denominator `d_a = 0`.
    LookupDenominator {
        /// The instance.
        instance: usize,
    },
    /// An instance's table-tree root has the denominator `d_t = 0`.
    TableDenominator {
        /// The instance.
        instance: usize,
    },
    /// An instance's roots have `n_a·d_t + n_t·d_a ≠ 0`: its lookups' fractions and its rows'
    /// fractions do not sum to zero.
    Unbalanced {
        /// The instance.
        instance: usize,
    },
    /// The values message holds `found` values where the instances whose tables have values
    /// need `expected`, one each.
    ValuesLength {
        /// The first instance with values whose value the message lacks, or the last instance
        /// when it holds more.
        instance: usize,
        /// One value per instance whose table has values.
        expected: usize,
        /// The values the message holds.
        found: usize,
    },
    /// An instance's lookup-tree leaf numerator claim is not 1.
    LeafNumerator {
        /// The instance.
        instance: usize,
    },
    /// An instance's table-tree leaf denominator claim is not `α − t̃(ρ_t)`.
    TableEvaluation {
        /// The instance.
        instance: usize,
    },
    /// The leaf stage has no LogUp instance, and the roots message, which must then be empty,
    /// holds `found` values. A fault of the proof that names no instance, so no table for
    /// `Reject(Lookup { table })`: the verifier's driver (#128) gives it its reason.
    RootsWithoutInstance {
        /// The values the message holds.
        found: usize,
    },
    /// The leaf stage has no LogUp instance, and the values message, which must then be empty,
    /// holds `found` values. A fault of the proof that names no instance, as for
    /// `RootsWithoutInstance`.
    ValuesWithoutInstance {
        /// The values the message holds.
        found: usize,
    },
    /// The caller could not evaluate an instance's table at its table tree's leaf point.
    TableUnavailable {
        /// The instance.
        instance: usize,
    },
    /// A result vector or claim point could not be reserved (`ERR-RES-MEMORY`).
    Memory,
    /// A label built here is not printable ASCII.
    Label(NotAscii),
    /// A challenge failed, or message encoding could not reserve memory (RFC-0004, SP-8).
    Transcript(TranscriptError),
}

impl LookupError {
    /// Whether the error is a fault of the proof, a rejection, rather than of the verifier,
    /// reported as Inconclusive: `ERR-RES-MEMORY` for exhausted memory,
    /// `ERR-INTERNAL` for other verifier faults.
    pub fn is_rejection(&self) -> bool {
        match self {
            LookupError::RootsLength { .. }
            | LookupError::LookupDenominator { .. }
            | LookupError::TableDenominator { .. }
            | LookupError::Unbalanced { .. }
            | LookupError::ValuesLength { .. }
            | LookupError::LeafNumerator { .. }
            | LookupError::TableEvaluation { .. }
            | LookupError::RootsWithoutInstance { .. }
            | LookupError::ValuesWithoutInstance { .. } => true,
            LookupError::TableUnavailable { .. }
            | LookupError::Memory
            | LookupError::Label(_)
            | LookupError::Transcript(_) => false,
        }
    }

    /// The instance whose table `Reject(Lookup { table })` (04 §2) names, for a fault of the
    /// proof that names one; `None` for `RootsWithoutInstance` and `ValuesWithoutInstance`, and
    /// for a fault of the verifier.
    pub fn rejected_instance(&self) -> Option<usize> {
        match *self {
            LookupError::RootsLength { instance, .. }
            | LookupError::LookupDenominator { instance }
            | LookupError::TableDenominator { instance }
            | LookupError::Unbalanced { instance }
            | LookupError::ValuesLength { instance, .. }
            | LookupError::LeafNumerator { instance }
            | LookupError::TableEvaluation { instance } => Some(instance),
            LookupError::RootsWithoutInstance { .. }
            | LookupError::ValuesWithoutInstance { .. }
            | LookupError::TableUnavailable { .. }
            | LookupError::Memory
            | LookupError::Label(_)
            | LookupError::Transcript(_) => None,
        }
    }
}

impl From<Failure> for LookupError {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Label(error) => LookupError::Label(error),
            Failure::Transcript(error) => LookupError::Transcript(error),
        }
    }
}

impl fmt::Display for LookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LookupError::RootsLength {
                instance,
                expected,
                found,
            } => write!(
                f,
                "the roots message holds {found} values where the instances need {expected}, wrong from instance {instance}"
            ),
            LookupError::LookupDenominator { instance } => write!(
                f,
                "instance {instance}: the lookup tree's root denominator is zero"
            ),
            LookupError::TableDenominator { instance } => write!(
                f,
                "instance {instance}: the table tree's root denominator is zero"
            ),
            LookupError::Unbalanced { instance } => write!(
                f,
                "instance {instance}: the lookups' and the rows' fractions do not sum to zero"
            ),
            LookupError::ValuesLength {
                instance,
                expected,
                found,
            } => write!(
                f,
                "the values message holds {found} values where the instances need {expected}, wrong from instance {instance}"
            ),
            LookupError::LeafNumerator { instance } => write!(
                f,
                "instance {instance}: the lookup tree's leaf numerator claim is not 1"
            ),
            LookupError::TableEvaluation { instance } => write!(
                f,
                "instance {instance}: the table tree's leaf denominator claim is not α − t̃(ρ)"
            ),
            LookupError::RootsWithoutInstance { found } => write!(
                f,
                "the roots message holds {found} values where the leaf stage has no LogUp instance"
            ),
            LookupError::ValuesWithoutInstance { found } => write!(
                f,
                "the values message holds {found} values where the leaf stage has no LogUp instance"
            ),
            LookupError::TableUnavailable { instance } => write!(
                f,
                "instance {instance}: the table could not be evaluated at its leaf point"
            ),
            LookupError::Memory => f.write_str("memory for LogUp claims could not be reserved"),
            LookupError::Label(error) => write!(f, "a LogUp label is not ASCII: {error}"),
            LookupError::Transcript(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for LookupError {}

/// Squeezes LogUp's challenges `α` and `β`, absorbs the roots message `roots` and checks the
/// roots of the leaf stage's `instances` LogUp instances, in schedule order; returns `α`, `β`
/// and each instance's root claims. The checks, in order, are those of the crate documentation
/// ('LogUp roots'): `α`, then `β`; the message holds `n_a, d_a, n_t, d_t` for each instance; it
/// is absorbed under `logup/roots`, even when empty; then instance by instance `d_a ≠ 0`,
/// `d_t ≠ 0` and `n_a·d_t + n_t·d_a = 0`.
///
/// The root claims are what each tree enters the levels with, at the level `H − h` where its
/// height `h` joins; a tree of height 0 joins none, and its root claims are its leaf claims. A
/// leaf stage without LogUp instance has `instances = 0`: the message must be empty.
pub fn check_roots(
    transcript: &mut Transcript,
    instances: usize,
    roots: &[Fp2],
) -> Result<Roots, LookupError> {
    check_roots_with(transcript, instances, roots)
}

/// [`check_roots`] on any transcript.
pub(crate) fn check_roots_with(
    transcript: &mut impl FiatShamir,
    instances: usize,
    roots: &[Fp2],
) -> Result<Roots, LookupError> {
    // 1. α, then β.
    let alpha = fiat_shamir::challenge_under(transcript, ALPHA)?;
    let beta = fiat_shamir::challenge_under(transcript, BETA)?;
    // 2. Four values per instance, checked before the message is encoded.
    let blocks = roots.chunks_exact(ROOTS_PER_INSTANCE);
    let complete = blocks.len();
    if complete != instances || !blocks.remainder().is_empty() {
        return Err(match instances.checked_sub(1) {
            Some(last) => LookupError::RootsLength {
                instance: complete.min(last),
                expected: instances.saturating_mul(ROOTS_PER_INSTANCE),
                found: roots.len(),
            },
            None => LookupError::RootsWithoutInstance { found: roots.len() },
        });
    }
    // 3. The message, absorbed whole, even when empty.
    fiat_shamir::absorb_under(transcript, ROOTS, roots)?;
    // 4. Instance by instance.
    let mut claims = reserve(complete)?;
    for (instance, block) in blocks.enumerate() {
        let &[n_a, d_a, n_t, d_t] = block else {
            // Not reached: `chunks_exact` gives blocks of exactly four.
            return Err(LookupError::RootsLength {
                instance,
                expected: instances.saturating_mul(ROOTS_PER_INSTANCE),
                found: roots.len(),
            });
        };
        if d_a == Fp2::ZERO {
            return Err(LookupError::LookupDenominator { instance });
        }
        if d_t == Fp2::ZERO {
            return Err(LookupError::TableDenominator { instance });
        }
        if n_a * d_t + n_t * d_a != Fp2::ZERO {
            return Err(LookupError::Unbalanced { instance });
        }
        claims.push(RootClaims {
            lookup: root(n_a, d_a),
            table: root(n_t, d_t),
        });
    }
    Ok(Roots {
        challenges: Challenges { alpha, beta },
        instances: claims,
    })
}

fn reserve<T>(len: usize) -> Result<Vec<T>, LookupError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| LookupError::Memory)?;
    Ok(values)
}

fn copy_point(point: &[Fp2]) -> Result<Vec<Fp2>, LookupError> {
    let mut values = reserve(point.len())?;
    values.extend_from_slice(point);
    Ok(values)
}

/// A tree's root `(n, d)` as its claims on its level 0, on no variable.
fn root(numerator: Fp2, denominator: Fp2) -> FractionClaim {
    FractionClaim {
        point: Vec::new(),
        numerator,
        denominator,
    }
}

/// Absorbs the values message `values` and runs each LogUp instance's leaf checks on `leaves`, in
/// schedule order; returns the claims they leave for later stages. The checks, in order, are those
/// of the crate documentation ('LogUp leaf checks'): the message holds one value per instance
/// whose table has values; it is absorbed under `logup/values`, even when empty; then instance by
/// instance the lookup tree's leaf numerator claim is 1, and the table tree's leaf denominator
/// claim is `α − t̃(ρ_t)`.
///
/// `table(i, ρ_t)` is instance `i`'s table polynomial `t̃ = kẽy_t + β·valũe_t`, its padding rows
/// copies of row 0, at its table tree's leaf point `ρ_t`, or `None` when the caller cannot
/// evaluate it, a fault of the verifier. It is asked only for the instances the checks reach, in
/// order. A leaf stage without LogUp instance passes no leaves: the message must be empty.
pub fn check_leaves<F>(
    transcript: &mut Transcript,
    challenges: &Challenges,
    leaves: &[Leaves],
    values: &[Fp2],
    table: F,
) -> Result<LeafClaims, LookupError>
where
    F: FnMut(usize, &[Fp2]) -> Option<Fp2>,
{
    check_leaves_with(transcript, challenges, leaves, values, table)
}

/// [`check_leaves`] on any transcript.
pub(crate) fn check_leaves_with<F>(
    transcript: &mut impl FiatShamir,
    challenges: &Challenges,
    leaves: &[Leaves],
    values: &[Fp2],
    mut table: F,
) -> Result<LeafClaims, LookupError>
where
    F: FnMut(usize, &[Fp2]) -> Option<Fp2>,
{
    let Challenges { alpha, beta } = *challenges;
    // 1. One value per instance whose table has values, checked before the message is encoded.
    let expected = leaves.iter().filter(|leaf| leaf.has_values).count();
    if values.len() != expected {
        let Some(last) = leaves.len().checked_sub(1) else {
            return Err(LookupError::ValuesWithoutInstance {
                found: values.len(),
            });
        };
        // A short message lacks the value of the instance with values that comes after the ones
        // it holds; a long one holds more than the last instance takes.
        let instance = leaves
            .iter()
            .enumerate()
            .filter(|(_, leaf)| leaf.has_values)
            .nth(values.len())
            .map_or(last, |(instance, _)| instance);
        return Err(LookupError::ValuesLength {
            instance,
            expected,
            found: values.len(),
        });
    }
    // 2. The message, absorbed whole, even when empty.
    fiat_shamir::absorb_under(transcript, VALUES, values)?;
    // 3. Instance by instance, each taking the next value if its table has values.
    let mut sent = values.iter();
    let mut claims = reserve(leaves.len())?;
    for (instance, leaf) in leaves.iter().enumerate() {
        let value = if leaf.has_values {
            // The count check above gives each such instance its value.
            Some(*sent.next().ok_or(LookupError::ValuesLength {
                instance,
                expected,
                found: values.len(),
            })?)
        } else {
            None
        };
        if leaf.lookup.numerator != Fp2::ONE {
            return Err(LookupError::LeafNumerator { instance });
        }
        let evaluation =
            table(instance, &leaf.table.point).ok_or(LookupError::TableUnavailable { instance })?;
        if leaf.table.denominator != alpha - evaluation {
            return Err(LookupError::TableEvaluation { instance });
        }
        let v = value.unwrap_or(Fp2::ZERO);
        claims.push(InstanceClaims {
            key: Claim {
                point: copy_point(&leaf.lookup.point)?,
                value: alpha - leaf.lookup.denominator - beta * v,
            },
            value: if let Some(value) = value {
                Some(Claim {
                    point: copy_point(&leaf.lookup.point)?,
                    value,
                })
            } else {
                None
            },
            multiplicity: Claim {
                point: copy_point(&leaf.table.point)?,
                value: -leaf.table.numerator,
            },
        });
    }
    Ok(LeafClaims { instances: claims })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::stub::{Op, Stub};
    use crate::fiat_shamir::tce;
    use tv_field::Fp;

    fn fp2(c0: u64, c1: u64) -> Fp2 {
        Fp2::new(Fp::reduce(c0), Fp::reduce(c1))
    }

    /// The stub's challenge number `n`.
    fn stub_challenge(n: u64) -> Fp2 {
        fp2(1000 + n, 77 * n + 5)
    }

    /// Balanced roots: `n_a/d_a + n_t/d_t = 0`, each instance's own.
    fn balanced(instances: u64) -> Vec<Fp2> {
        (0..instances)
            .flat_map(|i| {
                let (n_a, d_a, d_t) = (fp2(3 + i, 1), fp2(5, 2 + i), fp2(7 + i, 4));
                // n_t = −n_a·d_t/d_a.
                let n_t = -(n_a * d_t * d_a.inverse().unwrap());
                [n_a, d_a, n_t, d_t]
            })
            .collect()
    }

    /// RFC-0014 row 4: `α`, `β`, then the roots, absorbed whole as TCE `Vec<Fp2>`; and the root
    /// claims, on no variable.
    #[test]
    fn the_roots_run_the_transcript_schedule_in_order() {
        let roots = balanced(3);
        let mut stub = Stub::default();
        let checked = check_roots_with(&mut stub, 3, &roots).unwrap();
        assert_eq!(
            stub.script(),
            ["?logup/alpha", "?logup/beta", "+logup/roots"]
        );
        assert_eq!(
            stub.ops[2],
            Op::Absorb("logup/roots".into(), tce(&roots).unwrap())
        );
        assert_eq!(
            checked.challenges,
            Challenges {
                alpha: stub_challenge(0),
                beta: stub_challenge(1)
            }
        );
        let expected: Vec<RootClaims> = roots
            .chunks_exact(4)
            .map(|block| RootClaims {
                lookup: root(block[0], block[1]),
                table: root(block[2], block[3]),
            })
            .collect();
        assert_eq!(checked.instances, expected);
    }

    /// RFC-0019 step 7.1, check by check: a zero denominator on either side, with a numerator of
    /// zero so that the sum alone would pass, and a sum that is not zero; the first failing
    /// instance is the one reported, and the three checks run in order within it.
    #[test]
    fn each_root_check_rejects_its_own_failure() {
        let honest = balanced(3);
        let check = |roots: &[Fp2]| check_roots_with(&mut Stub::default(), 3, roots);
        assert!(check(&honest).is_ok());
        for instance in 0..3 {
            let at = 4 * instance;
            // (n_a, d_a) = (0, 0): the sum n_a·d_t + n_t·d_a is zero.
            let mut zero_lookup = honest.clone();
            zero_lookup[at] = Fp2::ZERO;
            zero_lookup[at + 1] = Fp2::ZERO;
            assert_eq!(
                check(&zero_lookup),
                Err(LookupError::LookupDenominator { instance })
            );
            // (n_t, d_t) = (0, 0): likewise.
            let mut zero_table = honest.clone();
            zero_table[at + 2] = Fp2::ZERO;
            zero_table[at + 3] = Fp2::ZERO;
            assert_eq!(
                check(&zero_table),
                Err(LookupError::TableDenominator { instance })
            );
            // Both denominators zero: the lookup side is checked first.
            let mut both = honest.clone();
            both[at + 1] = Fp2::ZERO;
            both[at + 3] = Fp2::ZERO;
            assert_eq!(
                check(&both),
                Err(LookupError::LookupDenominator { instance })
            );
            // Each value changed by one unbalances the sum.
            for k in 0..4 {
                let mut changed = honest.clone();
                changed[at + k] += Fp2::ONE;
                assert_eq!(
                    check(&changed),
                    Err(LookupError::Unbalanced { instance }),
                    "instance {instance}, value {k}"
                );
            }
        }
        // Two instances at fault: the first is reported.
        let mut two = honest.clone();
        two[4] += Fp2::ONE;
        two[11] = Fp2::ZERO;
        assert_eq!(check(&two), Err(LookupError::Unbalanced { instance: 1 }));
    }

    /// A roots message of another length is rejected after `α` and `β`, before it is encoded or
    /// absorbed, at the first instance whose four values it lacks, or at the last instance when it
    /// is too long.
    #[test]
    fn a_roots_message_of_another_length_is_rejected_before_it_is_absorbed() {
        let honest = balanced(3);
        for (found, instance) in [(0, 0), (3, 0), (4, 1), (11, 2), (13, 2), (16, 2), (1000, 2)] {
            let roots: Vec<Fp2> = honest.iter().copied().cycle().take(found).collect();
            let mut stub = Stub::default();
            assert_eq!(
                check_roots_with(&mut stub, 3, &roots),
                Err(LookupError::RootsLength {
                    instance,
                    expected: 12,
                    found
                }),
                "{found} values"
            );
            assert_eq!(stub.script(), ["?logup/alpha", "?logup/beta"]);
        }
        // A count no message can hold is still a typed error, its expected length saturated.
        assert_eq!(
            check_roots_with(&mut Stub::default(), usize::MAX, &honest),
            Err(LookupError::RootsLength {
                instance: 3,
                expected: usize::MAX,
                found: 12
            })
        );
    }

    /// `α` or `β` without a value stops the roots, a fault of the verifier.
    #[test]
    fn an_exhausted_alpha_or_beta_is_a_fault_of_the_verifier() {
        for fail_at in 0..2 {
            let mut stub = Stub::failing_at(fail_at);
            let error = check_roots_with(&mut stub, 3, &balanced(3)).unwrap_err();
            assert_eq!(error, LookupError::Transcript(TranscriptError::Exhausted));
            assert_eq!(error.rejected_instance(), None);
            assert_eq!(stub.ops.len(), fail_at + 1);
        }
    }

    /// Leaves that pass for `α = stub_challenge(0)`: instance `i` has the table evaluation
    /// `t_i`, and every instance but the second has values.
    fn honest_leaves(alpha: Fp2) -> (Vec<Leaves>, Vec<Fp2>) {
        let leaves = (0..3u64)
            .map(|i| Leaves {
                lookup: FractionClaim {
                    point: vec![fp2(i, 1); i as usize + 1],
                    numerator: Fp2::ONE,
                    denominator: fp2(40 + i, 3),
                },
                table: FractionClaim {
                    point: vec![fp2(i, 2); 2],
                    numerator: fp2(50 + i, 0),
                    denominator: alpha - table_eval(i as usize),
                },
                has_values: i != 1,
            })
            .collect();
        (leaves, vec![fp2(60, 1), fp2(62, 1)])
    }

    fn table_eval(instance: usize) -> Fp2 {
        fp2(70 + instance as u64, 8)
    }

    fn challenges() -> Challenges {
        Challenges {
            alpha: stub_challenge(0),
            beta: stub_challenge(1),
        }
    }

    /// RFC-0014 row 5 and RFC-0019 step 7.3: the values, absorbed whole, then the checks; and
    /// the claims left for later stages, the value taken in schedule order by the instances whose
    /// tables have values.
    #[test]
    fn the_leaf_checks_absorb_the_values_and_leave_their_claims() {
        let Challenges { alpha, beta } = challenges();
        let (leaves, values) = honest_leaves(alpha);
        let mut stub = Stub::default();
        let mut asked = Vec::new();
        let claims = check_leaves_with(&mut stub, &challenges(), &leaves, &values, |i, point| {
            asked.push((i, point.to_vec()));
            Some(table_eval(i))
        })
        .unwrap();
        assert_eq!(stub.script(), ["+logup/values"]);
        assert_eq!(
            stub.ops[0],
            Op::Absorb("logup/values".into(), tce(&values).unwrap())
        );
        let table_points: Vec<(usize, Vec<Fp2>)> = leaves
            .iter()
            .enumerate()
            .map(|(i, leaf)| (i, leaf.table.point.clone()))
            .collect();
        assert_eq!(asked, table_points);
        let v = [Some(values[0]), None, Some(values[1])];
        for (i, (leaf, claims)) in leaves.iter().zip(&claims.instances).enumerate() {
            let sent = v[i].unwrap_or(Fp2::ZERO);
            assert_eq!(
                claims.key,
                Claim {
                    point: leaf.lookup.point.clone(),
                    value: alpha - leaf.lookup.denominator - beta * sent,
                }
            );
            assert_eq!(
                claims.value,
                v[i].map(|value| Claim {
                    point: leaf.lookup.point.clone(),
                    value
                })
            );
            assert_eq!(
                claims.multiplicity,
                Claim {
                    point: leaf.table.point.clone(),
                    value: -leaf.table.numerator,
                }
            );
        }
    }

    /// Each leaf check rejects its own failure, at the instance that fails first; the numerator
    /// is checked before the table is evaluated.
    #[test]
    fn each_leaf_check_rejects_its_own_failure() {
        let alpha = challenges().alpha;
        let (honest, values) = honest_leaves(alpha);
        for instance in 0..3 {
            for numerator in [Fp2::ZERO, fp2(2, 0), fp2(1, 1)] {
                let mut leaves = honest.clone();
                leaves[instance].lookup.numerator = numerator;
                // The table of a later instance is never asked for.
                let outcome = check_leaves_with(
                    &mut Stub::default(),
                    &challenges(),
                    &leaves,
                    &values,
                    |i, _| (i < instance).then(|| table_eval(i)),
                );
                assert_eq!(outcome, Err(LookupError::LeafNumerator { instance }));
            }
            for change in [Fp2::ONE, -Fp2::ONE, fp2(0, 1)] {
                let mut leaves = honest.clone();
                leaves[instance].table.denominator += change;
                let outcome = check_leaves_with(
                    &mut Stub::default(),
                    &challenges(),
                    &leaves,
                    &values,
                    |i, _| Some(table_eval(i)),
                );
                assert_eq!(outcome, Err(LookupError::TableEvaluation { instance }));
            }
            // The table evaluation off by one, for the honest leaves.
            let outcome = check_leaves_with(
                &mut Stub::default(),
                &challenges(),
                &honest,
                &values,
                |i, _| {
                    let off = if i == instance { Fp2::ONE } else { Fp2::ZERO };
                    Some(table_eval(i) + off)
                },
            );
            assert_eq!(outcome, Err(LookupError::TableEvaluation { instance }));
            // A table the caller cannot evaluate is its fault.
            let outcome = check_leaves_with(
                &mut Stub::default(),
                &challenges(),
                &honest,
                &values,
                |i, _| (i != instance).then(|| table_eval(i)),
            );
            assert_eq!(outcome, Err(LookupError::TableUnavailable { instance }));
            assert_eq!(outcome.unwrap_err().rejected_instance(), None);
        }
    }

    /// A values message of another length is rejected before it is encoded or absorbed: a short
    /// one at the first instance with values whose value it lacks, a long one at the last
    /// instance.
    #[test]
    fn a_values_message_of_another_length_is_rejected_before_it_is_absorbed() {
        let alpha = challenges().alpha;
        let (leaves, values) = honest_leaves(alpha);
        // Instances 0 and 2 have values.
        for (found, instance) in [(0, 0), (1, 2), (3, 2), (1000, 2)] {
            let sent: Vec<Fp2> = values.iter().copied().cycle().take(found).collect();
            let mut stub = Stub::default();
            assert_eq!(
                check_leaves_with(&mut stub, &challenges(), &leaves, &sent, |i, _| Some(
                    table_eval(i)
                )),
                Err(LookupError::ValuesLength {
                    instance,
                    expected: 2,
                    found
                }),
                "{found} values"
            );
            assert!(stub.ops.is_empty());
        }
        // No instance has values: any value is one too many, at the last instance.
        let without: Vec<Leaves> = leaves
            .iter()
            .cloned()
            .map(|leaf| Leaves {
                has_values: false,
                ..leaf
            })
            .collect();
        assert_eq!(
            check_leaves_with(
                &mut Stub::default(),
                &challenges(),
                &without,
                &values[..1],
                |i, _| Some(table_eval(i))
            ),
            Err(LookupError::ValuesLength {
                instance: 2,
                expected: 0,
                found: 1
            })
        );
        let outcome = check_leaves_with(
            &mut Stub::default(),
            &challenges(),
            &without,
            &[],
            |i, _| Some(table_eval(i)),
        )
        .unwrap();
        assert!(
            outcome
                .instances
                .iter()
                .all(|claims| claims.value.is_none())
        );
    }

    /// RFC-0008 'Fraction tree': the roots and the values are absorbed even when empty. A leaf
    /// stage without LogUp instance squeezes `α` and `β`, absorbs an empty roots message and an
    /// empty values message, and leaves no claim; a message that holds values is the proof's
    /// fault, naming no instance, and is not absorbed.
    #[test]
    fn a_leaf_stage_without_instance_absorbs_its_empty_messages() {
        let mut stub = Stub::default();
        let roots = check_roots_with(&mut stub, 0, &[]).unwrap();
        assert!(roots.instances.is_empty());
        assert_eq!(roots.challenges, challenges());
        let claims =
            check_leaves_with(&mut stub, &roots.challenges, &[], &[], |_, _| None).unwrap();
        assert!(claims.instances.is_empty());
        assert_eq!(
            stub.ops,
            [
                Op::Challenge("logup/alpha".into()),
                Op::Challenge("logup/beta".into()),
                Op::Absorb("logup/roots".into(), tce(&[]).unwrap()),
                Op::Absorb("logup/values".into(), tce(&[]).unwrap()),
            ]
        );
        for found in [1, 4, 1000] {
            let mut stub = Stub::default();
            let error = check_roots_with(&mut stub, 0, &balanced(250)[..found]).unwrap_err();
            assert_eq!(error, LookupError::RootsWithoutInstance { found });
            assert!(error.is_rejection() && error.rejected_instance().is_none());
            assert_eq!(stub.script(), ["?logup/alpha", "?logup/beta"]);
            let mut stub = Stub::default();
            let values = vec![Fp2::ONE; found];
            let error =
                check_leaves_with(&mut stub, &challenges(), &[], &values, |_, _| None).unwrap_err();
            assert_eq!(error, LookupError::ValuesWithoutInstance { found });
            assert!(error.is_rejection() && error.rejected_instance().is_none());
            assert!(stub.ops.is_empty());
        }
    }

    #[test]
    fn faults_of_the_proof_name_an_instance_and_faults_of_the_verifier_none() {
        let proof_faults = [
            LookupError::RootsLength {
                instance: 4,
                expected: 8,
                found: 7,
            },
            LookupError::LookupDenominator { instance: 4 },
            LookupError::TableDenominator { instance: 4 },
            LookupError::Unbalanced { instance: 4 },
            LookupError::ValuesLength {
                instance: 4,
                expected: 2,
                found: 3,
            },
            LookupError::LeafNumerator { instance: 4 },
            LookupError::TableEvaluation { instance: 4 },
        ];
        for error in proof_faults {
            assert_eq!(error.rejected_instance(), Some(4), "{error}");
            assert!(error.is_rejection(), "{error}");
        }
        for error in [
            LookupError::RootsWithoutInstance { found: 1 },
            LookupError::ValuesWithoutInstance { found: 1 },
        ] {
            assert_eq!(error.rejected_instance(), None, "{error}");
            assert!(error.is_rejection(), "{error}");
        }
        let verifier_faults = [
            LookupError::TableUnavailable { instance: 4 },
            LookupError::Label(tv_hash::transcript::Ascii::new("\n").unwrap_err()),
            LookupError::Transcript(TranscriptError::Exhausted),
        ];
        for error in verifier_faults {
            assert_eq!(error.rejected_instance(), None, "{error}");
            assert!(!error.is_rejection(), "{error}");
        }
    }

    /// The four labels are printable ASCII, so `Label` never comes from them.
    #[test]
    fn the_labels_are_ascii() {
        for label in [ALPHA, BETA, ROOTS, VALUES] {
            assert!(tv_hash::transcript::Ascii::new(label).is_ok(), "{label}");
        }
    }
}
