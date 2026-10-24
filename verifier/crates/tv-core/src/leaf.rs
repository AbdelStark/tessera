// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The shared fraction-tree/product-forest loop of RFC-0019 step 7, protocol/1.
//!
//! [`Schedule`] contains verifier-owned public geometry, never proof-supplied tree ids or
//! roots. It checks the LogUp roots, joins trees at `H - height`, checks one mixed sumcheck
//! per level, and checks the LogUp leaves against independently evaluated public tables.
//! Forests use the last `s` challenges of the common root point, before the level's gamma.
//! Height-zero fraction roots are already their leaf claims and join no level.
//!
//! [`verify`](Schedule::verify) stops after `logup/values`, before the single `stage/0/split`
//! message. Its result retains every outstanding key/value/multiplicity and forest claim.
//! The driver must bind these through RFC-0045's lookup split, the gadget reductions and the
//! openings. Passing these checks alone certifies no statement.
//!
//! A leaf stage in token groups (RFC-0054) runs through [`grouped::Stage`] instead, on the layouts
//! of [`groups`].

pub mod grouped;
pub mod groups;
mod model;
pub mod split;
pub use model::{ForestShape, ModelSchedule};

use core::fmt;
use tv_field::Fp2;
use tv_hash::transcript::Transcript;

use crate::fiat_shamir::{self, FiatShamir};
use crate::logup::{self, LeafClaims, Leaves, LookupError};
use crate::sumcheck::{SumcheckError, SumcheckProof};
use crate::tree::{self, FractionClaim, LevelClaim, TreeClaim};

/// One tree in the verifier's fixed public schedule (RFC-0010 'Tree levels').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tree {
    /// Instance `instance`'s lookup tree, of `2^height` leaves after row-0 padding.
    Lookup {
        /// Consecutive LogUp instance id, in table order.
        instance: usize,
        /// Height, including zero for one lookup.
        height: u32,
        /// Whether this instance reads a table with values.
        has_values: bool,
    },
    /// Instance `instance`'s table tree, in table-row order.
    Table {
        /// Consecutive LogUp instance id, in table order.
        instance: usize,
        /// Height, including zero for one table row.
        height: u32,
    },
    /// A forest of `2^trees_log` product trees; its roots are verifier-owned public data.
    Forest {
        /// Positive height of each product tree.
        height: u32,
        /// Variables of the root table, the high coordinates of every forest claim.
        trees_log: u32,
    },
}

impl Tree {
    /// Height of the fraction tree or of each product tree in the forest.
    pub fn height(self) -> u32 {
        match self {
            Self::Lookup { height, .. }
            | Self::Table { height, .. }
            | Self::Forest { height, .. } => height,
        }
    }
}

/// A validated public tree schedule. Construct it from the independent model derivation;
/// accepting a proof's tree order or root values would make verification unsound (SP-7).
#[derive(Debug)]
pub struct Schedule {
    trees: Vec<Tree>,
    instances: Vec<(usize, usize)>,
    height: u32,
}

/// The three messages consumed here, borrowed from the decoded `Round2Response`.
#[derive(Clone, Copy, Debug)]
pub struct Proof<'a> {
    /// Four roots per LogUp instance, in table order.
    pub roots: &'a [Fp2],
    /// One shared sumcheck per level from the roots.
    pub levels: &'a [SumcheckProof],
    /// One value evaluation per instance whose table has values.
    pub values: &'a [Fp2],
}

/// A product forest's outstanding leaf claim, tied to its public schedule position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForestClaim {
    /// Position of the forest in [`Schedule`]'s input tree slice.
    pub tree: usize,
    /// Claim on the complete stacked leaf table, including padded heads/queries/layers.
    pub claim: TreeClaim,
}

/// The debts after the shared levels and LogUp leaf checks, never an acceptance verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "bind every lookup and forest claim to gadget reductions and commitment openings"]
pub struct Verified {
    /// Outstanding instance claims, in table order; multiplicities keep the table leaf point.
    pub lookups: LeafClaims,
    /// Outstanding forest claims, in public tree order.
    pub forests: Vec<ForestClaim>,
    /// RFC-0045's `P`: final line challenge, then final round challenges in reverse order.
    /// It is empty when no tree has a level.
    pub point: Vec<Fp2>,
}

/// Failure of the first shared-leaf check that fails, or of the verifier itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// `Reject(Malformed)`: not exactly one proof per public level, before transcript work.
    LevelCount {
        /// Tallest public tree height.
        expected: u32,
        /// Number of level proofs supplied.
        found: usize,
    },
    /// An existing LogUp root/leaf rejection or verifier error, with its instance intact.
    Lookup(LookupError),
    /// An existing tree-level failure, retaining its level and exact rejected round.
    Level {
        /// Level from the roots, starting at zero.
        level: u32,
        /// Failure of a root challenge or the batched level check.
        error: SumcheckError,
    },
    /// `Inconclusive(ERR-RES-MEMORY)`: bounded metadata could not be reserved.
    Memory,
    /// `Inconclusive(ERR-INTERNAL)`: invalid verifier-owned geometry, missing public root
    /// evaluation, or an inconsistent internal state. It never blames the proof.
    Internal,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LevelCount { expected, found } => write!(
                f,
                "the leaf stage has {found} level proofs, expected {expected}"
            ),
            Self::Lookup(error) => write!(f, "{error}"),
            Self::Level { level, error } => write!(f, "tree level {level}: {error}"),
            Self::Memory => {
                f.write_str("memory for shared leaf verification could not be reserved")
            }
            Self::Internal => f.write_str("inconsistent verifier-owned leaf metadata"),
        }
    }
}
impl std::error::Error for Error {}

fn reserved<T>(len: usize) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| Error::Memory)?;
    Ok(out)
}
fn copied<T: Copy>(values: &[T]) -> Result<Vec<T>, Error> {
    let mut out = reserved(values.len())?;
    out.extend_from_slice(values);
    Ok(out)
}

impl Schedule {
    /// Validate a verifier-derived schedule before any proof is read. Lookup and table
    /// trees each name every instance `0..n` exactly once. Forest heights are positive;
    /// every height and forest leaf-variable count is at most 63 (RFC-0008/0010).
    /// The caller's resource policy must bound the public tree count before building it.
    ///
    /// # Errors
    /// [`Error::Internal`] for invalid own metadata; [`Error::Memory`] on reservation failure.
    pub fn new(trees: &[Tree]) -> Result<Self, Error> {
        let count = trees
            .iter()
            .filter(|tree| matches!(tree, Tree::Lookup { .. }))
            .count();
        let mut positions = reserved(count)?;
        positions.resize(count, (None, None));
        let mut height = 0;
        for (position, &tree) in trees.iter().enumerate() {
            let h = tree.height();
            if h > 63 {
                return Err(Error::Internal);
            }
            height = height.max(h);
            let slot = match tree {
                Tree::Lookup { instance, .. } => {
                    &mut positions.get_mut(instance).ok_or(Error::Internal)?.0
                }
                Tree::Table { instance, .. } => {
                    &mut positions.get_mut(instance).ok_or(Error::Internal)?.1
                }
                Tree::Forest { trees_log, .. } => {
                    if h == 0 || h.checked_add(trees_log).is_none_or(|vars| vars > 63) {
                        return Err(Error::Internal);
                    }
                    continue;
                }
            };
            if slot.replace(position).is_some() {
                return Err(Error::Internal);
            }
        }
        let mut instances = reserved(count)?;
        for (lookup, table) in positions {
            instances.push((
                lookup.ok_or(Error::Internal)?,
                table.ok_or(Error::Internal)?,
            ));
        }
        Ok(Self {
            trees: copied(trees)?,
            instances,
            height,
        })
    }

    /// The validated trees in their public schedule order.
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Number of LogUp instances (one lookup and one table tree each).
    pub fn instances(&self) -> usize {
        self.instances.len()
    }
    /// Number of shared levels, the tallest public tree's height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Check the complete shared tree/LogUp sequence on one transcript.
    ///
    /// `forest_root(tree, point)` evaluates that forest's fixed PUBLIC root polynomial;
    /// it must never read a root from the proof. `table(instance, beta, point)` independently
    /// evaluates the canonical table polynomial `key + beta * value`. The callbacks' ids
    /// come only from the validated schedule, and points only from this transcript.
    ///
    /// # Errors
    /// [`Error::LevelCount`] first, then LogUp roots, levels in order, and LogUp leaves.
    /// Existing typed errors retain their proof rejection versus verifier-fault meaning.
    /// A callback without its own public data is a verifier fault, not a proof rejection.
    pub fn verify(
        &self,
        transcript: &mut Transcript,
        proof: Proof<'_>,
        forest_root: impl FnMut(usize, &[Fp2]) -> Option<Fp2>,
        table: impl FnMut(usize, Fp2, &[Fp2]) -> Option<Fp2>,
    ) -> Result<Verified, Error> {
        self.verify_with(transcript, proof, forest_root, table)
    }

    fn verify_with(
        &self,
        transcript: &mut impl FiatShamir,
        proof: Proof<'_>,
        mut forest_root: impl FnMut(usize, &[Fp2]) -> Option<Fp2>,
        mut table: impl FnMut(usize, Fp2, &[Fp2]) -> Option<Fp2>,
    ) -> Result<Verified, Error> {
        if proof.levels.len() != usize::try_from(self.height).map_err(|_| Error::Internal)? {
            return Err(Error::LevelCount {
                expected: self.height,
                found: proof.levels.len(),
            });
        }
        let roots = logup::check_roots_with(transcript, self.instances.len(), proof.roots)
            .map_err(Error::Lookup)?;
        let mut current = reserved(self.trees.len())?;
        for &tree in &self.trees {
            current.push(match tree {
                Tree::Lookup { instance, .. } | Tree::Table { instance, .. } => {
                    let root = roots.instances.get(instance).ok_or(Error::Internal)?;
                    let root = if matches!(tree, Tree::Lookup { .. }) {
                        &root.lookup
                    } else {
                        &root.table
                    };
                    Some(LevelClaim::Fraction(FractionClaim {
                        point: Vec::new(),
                        numerator: root.numerator,
                        denominator: root.denominator,
                    }))
                }
                Tree::Forest { .. } => None,
            });
        }
        let mut point = Vec::new();
        for (level, level_proof) in (0..self.height).zip(proof.levels) {
            let joins = |tree: Tree| self.height.checked_sub(tree.height()) == Some(level);
            let s_max = self
                .trees
                .iter()
                .filter_map(|&tree| match tree {
                    Tree::Forest { trees_log, .. } if joins(tree) => Some(trees_log),
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            let mut root_point = reserved(usize::try_from(s_max).map_err(|_| Error::Internal)?)?;
            for coordinate in 0..s_max {
                root_point.push(
                    fiat_shamir::challenge_under(
                        transcript,
                        &format!("tree/{level}/root/{coordinate}"),
                    )
                    .map_err(|failure| Error::Level {
                        level,
                        error: failure.into(),
                    })?,
                );
            }
            let mut positions = reserved(self.trees.len())?;
            let mut entering = reserved(self.trees.len())?;
            for (position, &tree) in self.trees.iter().enumerate() {
                if let Tree::Forest { trees_log, .. } = tree
                    && joins(tree)
                {
                    let start =
                        usize::try_from(s_max.checked_sub(trees_log).ok_or(Error::Internal)?)
                            .map_err(|_| Error::Internal)?;
                    let at = copied(root_point.get(start..).ok_or(Error::Internal)?)?;
                    let value = forest_root(position, &at).ok_or(Error::Internal)?;
                    *current.get_mut(position).ok_or(Error::Internal)? =
                        Some(LevelClaim::Product(TreeClaim { point: at, value }));
                }
                if self
                    .height
                    .checked_sub(tree.height())
                    .is_some_and(|start| start <= level)
                {
                    positions.push(position);
                    entering.push(
                        current
                            .get_mut(position)
                            .and_then(Option::take)
                            .ok_or(Error::Internal)?,
                    );
                }
            }
            let checked = tree::verify_level_with(transcript, level, &entering, level_proof)
                .map_err(|error| Error::Level { level, error })?;
            if checked.claims.len() != positions.len() {
                return Err(Error::Internal);
            }
            for (position, claim) in positions.into_iter().zip(checked.claims) {
                *current.get_mut(position).ok_or(Error::Internal)? = Some(claim);
            }
            if level.checked_add(1) == Some(self.height) {
                point = reserved(checked.point.len().checked_add(1).ok_or(Error::Internal)?)?;
                point.push(checked.rho);
                point.extend(checked.point.into_iter().rev());
            }
        }
        let mut leaves = reserved(self.instances.len())?;
        for &(lookup, table) in &self.instances {
            let has_values = match self.trees.get(lookup) {
                Some(Tree::Lookup { has_values, .. }) => *has_values,
                _ => return Err(Error::Internal),
            };
            let take = |current: &mut Vec<Option<LevelClaim>>, at| match current
                .get_mut(at)
                .and_then(Option::take)
            {
                Some(LevelClaim::Fraction(claim)) => Ok(claim),
                _ => Err(Error::Internal),
            };
            leaves.push(Leaves {
                lookup: take(&mut current, lookup)?,
                table: take(&mut current, table)?,
                has_values,
            });
        }
        let lookups = logup::check_leaves_with(
            transcript,
            &roots.challenges,
            &leaves,
            proof.values,
            |i, at| table(i, roots.challenges.beta, at),
        )
        .map_err(Error::Lookup)?;
        let mut forests = reserved(
            self.trees
                .len()
                .saturating_sub(self.instances.len().saturating_mul(2)),
        )?;
        for (position, claim) in current.into_iter().enumerate() {
            match claim {
                Some(LevelClaim::Product(claim)) => forests.push(ForestClaim {
                    tree: position,
                    claim,
                }),
                None => {}
                Some(LevelClaim::Fraction(_)) => return Err(Error::Internal),
            }
        }
        Ok(Verified {
            lookups,
            forests,
            point,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::stub::{Op, Stub};
    use tv_hash::transcript::TranscriptError;

    #[test]
    fn exhausted_alpha_beta_and_forest_root_challenges_stop_immediately() {
        let schedule = Schedule::new(&[Tree::Forest {
            height: 1,
            trees_log: 2,
        }])
        .unwrap();
        let levels = [SumcheckProof {
            rounds: vec![],
            final_evals: vec![],
        }];
        let proof = Proof {
            roots: &[],
            levels: &levels,
            values: &[],
        };
        for fail_at in 0..4 {
            let mut stub = Stub::failing_at(fail_at);
            let result = schedule.verify_with(
                &mut stub,
                proof,
                |_, _| panic!("root challenges must finish first"),
                |_, _, _| panic!("no table evaluation"),
            );
            let exhausted = TranscriptError::Exhausted;
            let expected = if fail_at < 2 {
                Error::Lookup(LookupError::Transcript(exhausted))
            } else {
                Error::Level {
                    level: 0,
                    error: SumcheckError::Transcript(exhausted),
                }
            };
            assert_eq!(result, Err(expected));
            let label = match fail_at {
                0 => "logup/alpha",
                1 => "logup/beta",
                2 => "tree/0/root/0",
                _ => "tree/0/root/1",
            };
            assert_eq!(stub.ops.last(), Some(&Op::Challenge(label.into())));
        }
    }
}
