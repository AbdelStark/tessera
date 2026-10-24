// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A grouped leaf stage's checks through the merge (RFC-0054 'The leaf stage with groups', 'The
//! table batch', 'The merge', 'Messages'), on the layouts of [`groups`]. It is a
//! stage-level check: it returns what later stages owe, never a verdict.
//!
//! # Specification
//!
//! [`Stage::new`] takes a [`Grouped`] layout of `G = 2^{g_T + g_L}` groups and the stage's table
//! trees ([`TableTree`]: one per LogUp instance, ascending `TableId`, each of height `⌈log2 T⌉`
//! for its `T` rows). It refuses its own metadata, [`Error::Internal`]: `G < 2`, a stage of one
//! group being RFC-0010's ([`Schedule`](super::Schedule)); tables that do not strictly ascend or a
//! table tree taller than 63; a group's instance whose table has no table tree; a table tree that
//! no group holds a lookup of. It derives every group once, keeping only its message counts.
//!
//! [`Stage::verify`] then runs these checks in this order on one transcript, and the first that
//! fails is the one reported (RFC-0054 'Messages': "the first failing check in the order above is
//! the one reported").
//!
//! **Shapes**, before the transcript moves. The grouping fixes every length below, so a mismatch
//! is the decoded `Round2Response`'s, before RFC-0019 step 7 (RFC-0036 'Shapes'):
//! [`Error::Count`], `Reject(Malformed { artifact, code: ERR-CODEC-NONCANONICAL })`, as the
//! ungrouped leaf stage's level count and split count are.
//!
//! 1. `LogUpProof.roots`, then `LogUpProof.values`, are empty: each group's are in its
//!    `GroupProof` (RFC-0054 'Messages').
//! 2. `TreeProof.groups` holds `G` entries.
//! 3. Group by group, from 0: `roots` holds two values, `(n, d)`, per lookup tree of the group;
//!    `levels` holds `L_h` sumchecks, `L_h` its tallest tree's height ([`Group::levels`]);
//!    `values` one value per instance whose table has values; `splits` one value per entry of
//!    [`Group::split`].
//! 4. `table_roots` holds two values, `(n_t, d_t)`, per table tree; `TreeProof.levels` holds `H_T`
//!    sumchecks, the tallest table tree's height.
//!
//! A sumcheck's own rounds and evaluations are not shapes: they are the checks of its level or of
//! the merge below (RFC-0010 'Sumcheck messages'). `TreeProof.splits`, the atom splits of step 5,
//! is the caller's, under `stage/0/split` after this check.
//!
//! **Step 1, the challenges** (RFC-0014 'Transcript schedule', row 4 with groups): `α` under
//! `logup/alpha`, `β` under `logup/beta`, then `logup/roots` and `logup/values` absorbed empty.
//!
//! **Step 2, each group `h` from 0**, its schedule its lookup trees in table order, then its
//! forest shares in the stage's forest order ([`Group::instances`], [`Group::forests`]):
//!
//! 5. Its roots absorbed under `group/{h}/logup/roots`; tree by tree in schedule order, `d ≠ 0`,
//!    else [`Error::Denominator`], `Reject(Lookup { table })`.
//! 6. Its levels `ℓ = 0 … L_h − 1`, RFC-0010 'Tree levels' under `group/{h}/tree/{ℓ}/…`: the
//!    shares joining at `ℓ` take the last `s` of the challenges `group/{h}/tree/{ℓ}/root/{c}`,
//!    their root claim the verifier's own public root at the forest's root point
//!    ([`Share::root_point`](super::groups::Share::root_point)); each lookup tree enters at
//!    `L_h − h_a` with its root. A failure is [`Error::Level`] with `group: h`,
//!    `Reject(TreeLevel { group: h, level: ℓ, round })`.
//! 7. Its values absorbed under `group/{h}/logup/values`; instance by instance, the lookup tree's
//!    leaf numerator claim is 1, else [`Error::LeafNumerator`], `Reject(Lookup { table })`. Each
//!    leaves `kẽy(ρ_a) = α − q − β·v` and `valũe(ρ_a) = v` (RFC-0008 'Fraction tree').
//! 8. Its split values absorbed under `group/{h}/split`; instance by instance in table order,
//!    RFC-0045's identity over the group's blocks at the group's leaf point `P_h`, the key claim
//!    then the value claim: `Σ_b eq(P_h[k_b..h], o_b / 2^{k_b})·form_b + (1 − LT(P_h[0..h], m))·t_0`.
//!    A mismatch is [`Error::Split`], `Reject(TreeLevel { group: h, level: L_h, round: 0 })`.
//! 9. `γ_h` squeezed under `group/{h}/merge/gamma`.
//!
//! Group `h`'s claims, in order `k`, are its split values, then its shares' leaf claims: claim
//! `k` is `P̃_i(z ‖ h_S) = v_{k,h}` on merged polynomial `i`, `z` its coordinates outside `S_i`
//! (a split value's `P_h` coordinates, a share's leaf point) and `h_S` its bits on `S_i`.
//!
//! **Step 3, the table batch:**
//!
//! 10. `table_roots` absorbed under `tables/logup/roots`; table by table, `d_t ≠ 0`, else
//!     [`Error::TableDenominator`], `Reject(Lookup { table })`.
//! 11. Table by table, the identity: its groups' roots in group order, then its table root, summed
//!     by `(n, d) ⊕ (n', d') = (n·d' + n'·d, d·d')`, have numerator 0, else [`Error::Unbalanced`],
//!     `Reject(Lookup { table })`.
//! 12. The table trees' levels, RFC-0010's under `tables/tree/{ℓ}/…`, each tree entering at
//!     `H_T − h_t` with its root. A failure is [`Error::Level`] with `group: G`.
//! 13. Table by table, the table tree's leaf denominator claim is `α − t̃(ρ_t)`, `t̃ = kẽy_t +
//!     β·valũe_t` evaluated by the caller, else [`Error::TableEvaluation`], `Reject(Lookup {
//!     table })`. It leaves `m̃(ρ_t) = −p_t`, the multiplicity claim.
//!
//! **Step 4, the merge** (RFC-0054 'The merge'), over the merged polynomials of
//! [`Grouped::polynomials`], `n_max` the most variables among them:
//!
//! 14. `β_m` squeezed under `merge/beta`.
//! 15. The claim `Σ_h β_m^h·Σ_k γ_h^k·2^{n_max − n_i}·v_{k,h}`; the rounds of degree 2 over `n_max`
//!     variables under `merge/round/{j}` and `merge/r/{j}` ([`sumcheck::reduce`], `p(0), p(2)`
//!     per round). Their failures are [`Error::Merge`], `Reject(LeafMerge { round })`.
//! 16. One evaluation per merged polynomial, absorbed under `merge/evals`, else
//!     `EvaluationCount`, `LeafMerge { round: n_max }`.
//! 17. `Σ_i W_i(r_i)·P̃_i(r_i)` equals the last claim, else `FinalMismatch`, `LeafMerge { round:
//!     n_max }`. `r_i` is the last `n_i` challenges, binding `P_i`'s variables outside `S_i`
//!     first, lowest first, then those in `S_i`, lowest first; the verifier computes every
//!     `W_i(r_i) = Σ_h β_m^h·γ_h^k·eq(z_{i,h}, r_i[..n_i − |S_i|])·eq(h_S, r_i[n_i − |S_i|..])`
//!     itself, over the groups that claim `P_i`, its claim there the `k`-th.
//!
//! **The result** ([`Merged`]): each merged polynomial's claim `P̃_i(r_i)` at its own point, in the
//! polynomial's natural variable order (variable `v` outside `S_i` takes the `v`-th coordinate of
//! `r_i`'s first part, counted among those outside `S_i`, and the `j`-th variable of `S_i` the
//! `j`-th of its second part): the tensors in the ungrouped split's order, then each forest's
//! leaf table; and each table's multiplicity claim. Nothing in it is verified until the stages
//! and openings that take it check it.
//!
//! **Memory.** A verifier holds one group's layout at a time: [`Stage::verify`] derives each
//! group's twice, for its checks and for the merge's weights, and keeps of each group only `γ_h`,
//! `P_h`, its shares' leaf claims and its share of the merge's claim. `W_i(r_i)` costs one `eq`
//! product per claim.
//!
//! **Faults of the verifier.** A callback without its public data, a label that is not ASCII, an
//! inconsistent own layout or a challenge without value is `Inconclusive`: [`Error::Memory`]
//! (`ERR-RES-MEMORY`) or [`Error::Internal`] (`ERR-INTERNAL`), or a transcript failure inside
//! [`Error::Level`] or [`Error::Merge`] that names no round.

use core::fmt;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Transcript, TranscriptError};
use tv_tables::TableId;

use super::groups::{self, Group, Grouped, Polynomial, Source};
use super::split::ColumnClaim;
use super::{ForestClaim, copied, reserved};
use crate::fiat_shamir::{self, Failure, FiatShamir, Step};
use crate::logup::Claim;
use crate::protocol::{GroupProof, LogUpProof, TreeProof};
use crate::sumcheck::{self, Prefix, SumcheckError, SumcheckProof};
use crate::tree::{self, FractionClaim, LevelClaim, TreeClaim};
use crate::verdict::{ArtifactKind, ErrorCode, RejectReason, Verdict, VerifierError};

/// The merge's degree: `W_i·P̃_i` is of degree 2 in each variable (RFC-0054 'Rounds').
pub const MERGE_DEGREE: usize = 2;

/// A table tree of the stage's table batch: one per LogUp instance (RFC-0054 'The table batch').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableTree {
    /// The table, which names the instance.
    pub table: TableId,
    /// `h_t = ⌈log2 T⌉` for its `T` rows.
    pub height: u32,
}

/// The messages a grouped stage consumes, borrowed from the decoded `Round2Response`.
#[derive(Clone, Copy, Debug)]
pub struct Proof<'a> {
    /// The ungrouped stage's roots and values, both empty in a grouped stage.
    pub logup: &'a LogUpProof,
    /// The table batch's levels, the groups, the table roots and the merge. Its `splits`, the
    /// atom splits, are not read here.
    pub trees: &'a TreeProof,
}

/// The message whose length the grouping fixes (module documentation, 'Shapes').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Message {
    /// `LogUpProof.roots`, empty.
    LogUpRoots,
    /// `LogUpProof.values`, empty.
    LogUpValues,
    /// `TreeProof.groups`, `G` entries.
    Groups,
    /// Group `group`'s `roots`.
    Roots {
        /// `h`.
        group: u32,
    },
    /// Group `group`'s `levels`.
    Levels {
        /// `h`.
        group: u32,
    },
    /// Group `group`'s `values`.
    Values {
        /// `h`.
        group: u32,
    },
    /// Group `group`'s `splits`.
    Splits {
        /// `h`.
        group: u32,
    },
    /// `TreeProof.table_roots`.
    TableRoots,
    /// `TreeProof.levels`, the table batch's.
    TableLevels,
}

/// The first check of the module documentation that failed, or a fault of the verifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Checks 1 to 4, `Reject(Malformed { artifact, code: ERR-CODEC-NONCANONICAL })`.
    Count {
        /// The message.
        message: Message,
        /// The length the grouping fixes.
        expected: usize,
        /// The message's length.
        found: usize,
    },
    /// Check 5, `Reject(Lookup { table })`: a group's lookup-tree root has `d = 0`.
    Denominator {
        /// `h`.
        group: u32,
        /// The tree's table.
        table: TableId,
    },
    /// Checks 6 and 12: a level failed, or its transcript did. `Reject(TreeLevel { group, level,
    /// round })` with the round of [`SumcheckError::rejected_round`]; a fault of the verifier
    /// when it has none.
    Level {
        /// `h` for group `h`'s levels, `G` for the table batch's.
        group: u32,
        /// `ℓ`, from the batch's roots.
        level: u32,
        /// The level's failure.
        error: SumcheckError,
    },
    /// Check 7, `Reject(Lookup { table })`: a group's lookup-tree leaf numerator claim is not 1.
    LeafNumerator {
        /// `h`.
        group: u32,
        /// The tree's table.
        table: TableId,
    },
    /// Check 8, `Reject(TreeLevel { group, level, round: 0 })`: RFC-0045's identity fails.
    Split {
        /// `h`.
        group: u32,
        /// `L_h`, the group's level count.
        level: u32,
    },
    /// Check 10, `Reject(Lookup { table })`: a table-tree root has `d_t = 0`.
    TableDenominator {
        /// The table.
        table: TableId,
    },
    /// Check 11, `Reject(Lookup { table })`: the groups' and the table's fractions do not sum to
    /// zero.
    Unbalanced {
        /// The table.
        table: TableId,
    },
    /// Check 13, `Reject(Lookup { table })`: the table tree's leaf denominator claim is not
    /// `α − t̃(ρ_t)`.
    TableEvaluation {
        /// The table.
        table: TableId,
    },
    /// Checks 15 to 17: `Reject(LeafMerge { round })` with the round of
    /// [`SumcheckError::rejected_round`]; a fault of the verifier when it has none.
    Merge(SumcheckError),
    /// A challenge outside a level or the merge found no value: `Inconclusive`.
    Transcript(TranscriptError),
    /// `Inconclusive(ERR-RES-MEMORY)`.
    Memory,
    /// `Inconclusive(ERR-INTERNAL)`: own metadata, a callback without its public data, or an
    /// inconsistent internal state. It never blames the proof.
    Internal,
}

impl Error {
    /// The reason of `Reject` (04 §2) for a fault of the proof, `artifact` being the outer
    /// artifact a `Malformed` names (`Round2Response` in live mode, `NiProof` otherwise); `None`
    /// for a fault of the verifier.
    pub fn rejection(&self, artifact: ArtifactKind) -> Option<RejectReason> {
        match *self {
            Self::Count { .. } => Some(RejectReason::Malformed {
                artifact,
                code: ErrorCode::CodecNoncanonical,
            }),
            Self::Denominator { table, .. }
            | Self::LeafNumerator { table, .. }
            | Self::TableDenominator { table }
            | Self::Unbalanced { table }
            | Self::TableEvaluation { table } => Some(RejectReason::Lookup { table }),
            Self::Level {
                group,
                level,
                error,
            } => error.rejected_round().map(|round| RejectReason::TreeLevel {
                group,
                level,
                round,
            }),
            Self::Split { group, level } => Some(RejectReason::TreeLevel {
                group,
                level,
                round: 0,
            }),
            Self::Merge(error) => error
                .rejected_round()
                .map(|round| RejectReason::LeafMerge { round }),
            Self::Transcript(_) | Self::Memory | Self::Internal => None,
        }
    }

    /// The verdict this error ends verification with: `Reject` with [`rejection`](Self::rejection)
    /// for a fault of the proof, else `Inconclusive` with `ERR-RES-MEMORY` for exhausted memory,
    /// the transcript's included, and `ERR-INTERNAL` otherwise.
    pub fn verdict(&self, artifact: ArtifactKind) -> Verdict {
        if let Some(reason) = self.rejection(artifact) {
            return Verdict::Reject(reason);
        }
        let memory = match *self {
            Self::Memory => true,
            Self::Transcript(error)
            | Self::Level {
                error: SumcheckError::Transcript(error),
                ..
            }
            | Self::Merge(SumcheckError::Transcript(error)) => error == TranscriptError::Memory,
            _ => false,
        };
        Verdict::Inconclusive(VerifierError {
            code: if memory {
                ErrorCode::Memory
            } else {
                ErrorCode::Internal
            },
        })
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count {
                message,
                expected,
                found,
            } => write!(
                f,
                "{message:?} holds {found} values where the grouping fixes {expected}"
            ),
            Self::Denominator { group, table } => write!(
                f,
                "group {group}: the lookup tree of table {} has a zero root denominator",
                table.0
            ),
            Self::Level {
                group,
                level,
                error,
            } => write!(f, "group {group}, tree level {level}: {error}"),
            Self::LeafNumerator { group, table } => write!(
                f,
                "group {group}: the lookup tree of table {} has a leaf numerator claim other than 1",
                table.0
            ),
            Self::Split { group, .. } => {
                write!(f, "group {group}: the split does not give the leaf claims")
            }
            Self::TableDenominator { table } => write!(
                f,
                "the table tree of table {} has a zero root denominator",
                table.0
            ),
            Self::Unbalanced { table } => write!(
                f,
                "table {}: the groups' and the rows' fractions do not sum to zero",
                table.0
            ),
            Self::TableEvaluation { table } => write!(
                f,
                "table {}: the table tree's leaf denominator claim is not α − t̃(ρ)",
                table.0
            ),
            Self::Merge(error) => write!(f, "the leaf merge: {error}"),
            Self::Transcript(error) => write!(f, "{error}"),
            Self::Memory => f.write_str("memory for the grouped leaf stage could not be reserved"),
            Self::Internal => f.write_str("inconsistent verifier-owned grouped leaf metadata"),
        }
    }
}
impl std::error::Error for Error {}

/// What later stages owe after a grouped leaf stage (module documentation, 'The result'): never
/// an acceptance verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "bind every merged claim to the reduction stages and every multiplicity to the openings"]
pub struct Merged {
    /// One claim per tensor, site and real layer, in the ungrouped split's order, each at its own
    /// point `r_i` in its natural variable order: RFC-0037's per-layer value for that tensor.
    pub columns: Vec<ColumnClaim>,
    /// One claim per forest leaf table, in the stage's forest order, `tree` its schedule
    /// position: the leaf claim stage 1 consumes.
    pub forests: Vec<ForestClaim>,
    /// Each table's multiplicity claim from the table batch, in table order, at its table tree's
    /// leaf point.
    pub multiplicities: Vec<(TableId, Claim)>,
    /// The merge's challenges, in round order.
    pub point: Vec<Fp2>,
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    roots: usize,
    levels: u32,
    values: usize,
    splits: usize,
}

/// A grouped leaf stage's validated public schedule (module documentation).
#[derive(Debug)]
pub struct Stage<'a> {
    grouped: &'a Grouped,
    tables: Vec<TableTree>,
    shapes: Vec<Shape>,
    /// `H_T`.
    height: u32,
    n_max: u32,
}

/// What group `h` leaves for the merge: `γ_h`, `P_h`, its shares' leaf claims, and
/// `Σ_k γ_h^k·2^{n_max − n_i}·v_{k,h}`.
struct Kept {
    gamma: Fp2,
    point: Vec<Fp2>,
    forests: Vec<TreeClaim>,
    claimed: Fp2,
}

fn vector<T>(n: usize) -> Result<Vec<T>, Error> {
    reserved(n).map_err(own)
}
fn own(error: super::Error) -> Error {
    match error {
        super::Error::Memory => Error::Memory,
        _ => Error::Internal,
    }
}
fn layout(error: groups::Error) -> Error {
    match error {
        groups::Error::Memory => Error::Memory,
        _ => Error::Internal,
    }
}
fn width(n: u32) -> Result<usize, Error> {
    usize::try_from(n).map_err(|_| Error::Internal)
}
fn count(message: Message, expected: usize, found: usize) -> Result<(), Error> {
    if expected == found {
        Ok(())
    } else {
        Err(Error::Count {
            message,
            expected,
            found,
        })
    }
}
/// A label built from RFC-0014's patterns, fixed or decimal: never a fault in practice.
fn absorb(transcript: &mut impl FiatShamir, label: &str, values: &[Fp2]) -> Result<(), Error> {
    fiat_shamir::absorb_under(transcript, label, values).map_err(|_| Error::Internal)
}
fn challenge(transcript: &mut impl FiatShamir, label: &str) -> Result<Fp2, Error> {
    fiat_shamir::challenge_under(transcript, label).map_err(|failure| match failure {
        Failure::Transcript(error) => Error::Transcript(error),
        Failure::Label(_) => Error::Internal,
    })
}
/// `(n, d) ⊕ (n', d') = (n·d' + n'·d, d·d')` (RFC-0008 'Fraction tree').
fn add((n, d): (Fp2, Fp2), (m, e): (Fp2, Fp2)) -> (Fp2, Fp2) {
    (n * e + m * d, d * e)
}
/// `eq` on one coordinate, `1 − a − b + 2ab`.
fn eq1(a: Fp2, b: Fp2) -> Fp2 {
    let ab = a * b;
    Fp2::ONE - a - b + ab + ab
}
/// `Π_j eq(bit j of bits, r_j)`.
fn eq_bits(bits: u32, point: &[Fp2]) -> Result<Fp2, Error> {
    let mut out = Fp2::ONE;
    for (j, &r) in point.iter().enumerate() {
        let on = bits
            .checked_shr(u32::try_from(j).map_err(|_| Error::Internal)?)
            .unwrap_or(0)
            & 1;
        out *= if on == 1 { r } else { Fp2::ONE - r };
    }
    Ok(out)
}
/// The last `n` coordinates of `point`.
fn suffix(point: &[Fp2], n: u32) -> Result<&[Fp2], Error> {
    point
        .len()
        .checked_sub(width(n)?)
        .and_then(|start| point.get(start..))
        .ok_or(Error::Internal)
}
/// `2^e` in `Fp`.
fn two_to(e: u32) -> Fp {
    Fp::reduce(2).pow(u64::from(e))
}

/// One tree of a batch's schedule as it enters the levels.
enum Entry {
    /// A fraction tree, with its root claims.
    Fraction { height: u32, root: FractionClaim },
    /// A forest share, whose root claim is public.
    Forest { height: u32, trees_log: u32 },
}
impl Entry {
    fn height(&self) -> u32 {
        match *self {
            Self::Fraction { height, .. } | Self::Forest { height, .. } => height,
        }
    }
}
/// A batch's leaf claims in schedule order, and its leaf point `P`.
struct Leveled {
    claims: Vec<Option<LevelClaim>>,
    point: Vec<Fp2>,
}
enum Halt {
    Level { level: u32, error: SumcheckError },
    Fault(Error),
}
impl From<Error> for Halt {
    fn from(error: Error) -> Self {
        Self::Fault(error)
    }
}
impl Halt {
    fn of(self, group: u32) -> Error {
        match self {
            Self::Level { level, error } => Error::Level {
                group,
                level,
                error,
            },
            Self::Fault(error) => error,
        }
    }
}

/// RFC-0010 'Tree levels' for one batch under its prefix: a forest share joining at `ℓ` takes the
/// last `s` of the `{prefix}/root/{c}` challenges, its claim `root(entry, point)`, public; a
/// fraction tree enters at `H − h` with its root; `P` is the last level's line challenge, then its
/// round challenges from the last.
fn levels(
    transcript: &mut impl FiatShamir,
    prefix: impl Fn(u32) -> Prefix,
    entries: Vec<Entry>,
    proofs: &[SumcheckProof],
    mut root: impl FnMut(usize, &[Fp2]) -> Option<Fp2>,
) -> Result<Leveled, Halt> {
    let height = entries.iter().map(Entry::height).max().unwrap_or(0);
    if proofs.len() != width(height)? {
        return Err(Error::Internal.into());
    }
    let mut heights = vector(entries.len())?;
    let mut forests = vector(entries.len())?;
    let mut current = vector(entries.len())?;
    for entry in entries {
        heights.push(entry.height());
        match entry {
            Entry::Fraction { root, .. } => {
                forests.push(None);
                current.push(Some(LevelClaim::Fraction(root)));
            }
            Entry::Forest { trees_log, .. } => {
                forests.push(Some(trees_log));
                current.push(None);
            }
        }
    }
    let mut point = Vec::new();
    for (level, proof) in (0..height).zip(proofs) {
        let prefix = prefix(level);
        let joins = |h: u32| height.checked_sub(h) == Some(level);
        let s_max = heights
            .iter()
            .zip(&forests)
            .filter_map(|(&h, &s)| s.filter(|_| joins(h)))
            .max()
            .unwrap_or(0);
        let mut root_point = vector(width(s_max)?)?;
        for c in 0..width(s_max)? {
            root_point.push(
                fiat_shamir::challenge(transcript, prefix, Step::Root(c))
                    .map_err(|error| Halt::Level { level, error })?,
            );
        }
        let mut positions = vector(heights.len())?;
        let mut entering = vector(heights.len())?;
        for (position, (&h, &s)) in heights.iter().zip(&forests).enumerate() {
            if let Some(s) = s
                && joins(h)
            {
                let at = root_point
                    .get(width(s_max.checked_sub(s).ok_or(Error::Internal)?)?..)
                    .ok_or(Error::Internal)?;
                let value = root(position, at).ok_or(Error::Internal)?;
                *current.get_mut(position).ok_or(Error::Internal)? =
                    Some(LevelClaim::Product(TreeClaim {
                        point: copied(at).map_err(own)?,
                        value,
                    }));
            }
            if height.checked_sub(h).is_some_and(|start| start <= level) {
                positions.push(position);
                entering.push(
                    current
                        .get_mut(position)
                        .and_then(Option::take)
                        .ok_or(Error::Internal)?,
                );
            }
        }
        let checked = tree::verify_level_under(transcript, prefix, &entering, proof)
            .map_err(|error| Halt::Level { level, error })?;
        if checked.claims.len() != positions.len() {
            return Err(Error::Internal.into());
        }
        for (position, claim) in positions.into_iter().zip(checked.claims) {
            *current.get_mut(position).ok_or(Error::Internal)? = Some(claim);
        }
        if level.checked_add(1) == Some(height) {
            point = vector(checked.point.len().checked_add(1).ok_or(Error::Internal)?)?;
            point.push(checked.rho);
            point.extend(checked.point.into_iter().rev());
        }
    }
    Ok(Leveled {
        claims: current,
        point,
    })
}

impl<'a> Stage<'a> {
    /// Validate the stage's own metadata and record each group's message counts (module
    /// documentation). `tables` is verifier-derived: one table tree per instance, ascending.
    ///
    /// # Errors
    /// [`Error::Internal`] for metadata it refuses, [`Error::Memory`] on failed reservation.
    pub fn new(grouped: &'a Grouped, tables: &[TableTree]) -> Result<Self, Error> {
        let count = grouped.count();
        if count < 2
            || tables.iter().any(|t| t.height > 63)
            || tables
                .windows(2)
                .any(|w| matches!(w, [a, b] if a.table >= b.table))
        {
            return Err(Error::Internal);
        }
        let mut held = vector(tables.len())?;
        held.resize(tables.len(), false);
        let mut shapes = vector(width(count)?)?;
        for h in 0..count {
            let group = grouped.group(h).map_err(layout)?;
            for instance in &group.instances {
                let at = tables
                    .binary_search_by_key(&instance.table, |t| t.table)
                    .map_err(|_| Error::Internal)?;
                *held.get_mut(at).ok_or(Error::Internal)? = true;
            }
            shapes.push(Shape {
                roots: group
                    .instances
                    .len()
                    .checked_mul(2)
                    .ok_or(Error::Internal)?,
                levels: group.levels(),
                values: group.instances.iter().filter(|i| i.has_values).count(),
                splits: group.split.len(),
            });
        }
        if held.contains(&false) {
            return Err(Error::Internal);
        }
        let n_max = grouped
            .polynomials()
            .iter()
            .map(|p| p.variables)
            .max()
            .unwrap_or(0);
        if n_max > 63 {
            return Err(Error::Internal);
        }
        Ok(Self {
            grouped,
            tables: copied(tables).map_err(own)?,
            shapes,
            height: tables.iter().map(|t| t.height).max().unwrap_or(0),
            n_max,
        })
    }

    /// `G`, the groups.
    pub fn groups(&self) -> u32 {
        self.grouped.count()
    }

    /// `n_max`, the merge's rounds.
    pub fn rounds(&self) -> u32 {
        self.n_max
    }

    /// `H_T`, the table batch's levels.
    pub fn table_levels(&self) -> u32 {
        self.height
    }

    /// Run checks 1 to 17 of the module documentation on one transcript, in order.
    ///
    /// `forest_root(tree, point)` evaluates the PUBLIC root polynomial of the forest at schedule
    /// position `tree`, at its root point (queries, heads, layers), as
    /// [`ModelSchedule::forest_root`](super::ModelSchedule::forest_root) does; it never reads a
    /// root from the proof. `table(table, β, point)` evaluates `t̃ = kẽy_t + β·valũe_t`, its
    /// padding rows copies of row 0, at a table tree's leaf point. `row_zero(table)` gives row 0's
    /// key and, exactly for a table with values, its value. Their arguments come from the
    /// validated layout and the transcript only; `None` is a fault of the verifier.
    ///
    /// # Errors
    /// The first failing check's [`Error`]; [`Error::verdict`] maps it.
    pub fn verify(
        &self,
        transcript: &mut Transcript,
        proof: Proof<'_>,
        forest_root: impl FnMut(usize, &[Fp2]) -> Option<Fp2>,
        table: impl FnMut(TableId, Fp2, &[Fp2]) -> Option<Fp2>,
        row_zero: impl FnMut(TableId) -> Option<(Fp, Option<Fp>)>,
    ) -> Result<Merged, Error> {
        self.verify_with(transcript, proof, forest_root, table, row_zero)
    }

    fn verify_with(
        &self,
        transcript: &mut impl FiatShamir,
        proof: Proof<'_>,
        mut forest_root: impl FnMut(usize, &[Fp2]) -> Option<Fp2>,
        mut table: impl FnMut(TableId, Fp2, &[Fp2]) -> Option<Fp2>,
        mut row_zero: impl FnMut(TableId) -> Option<(Fp, Option<Fp>)>,
    ) -> Result<Merged, Error> {
        self.shapes(proof)?;
        // Step 1.
        let alpha = challenge(transcript, "logup/alpha")?;
        let beta = challenge(transcript, "logup/beta")?;
        absorb(transcript, "logup/roots", &[])?;
        absorb(transcript, "logup/values", &[])?;
        // Step 2. Each table's groups' fractions, summed in group order.
        let mut sums: Vec<Option<(Fp2, Fp2)>> = vector(self.tables.len())?;
        sums.resize(self.tables.len(), None);
        let mut kept = vector(self.shapes.len())?;
        for (h, group_proof) in (0..self.groups()).zip(&proof.trees.groups) {
            let group = self.grouped.group(h).map_err(layout)?;
            kept.push(self.group(
                transcript,
                &group,
                group_proof,
                (alpha, beta),
                &mut sums,
                &mut forest_root,
                &mut row_zero,
            )?);
        }
        // Step 3.
        let multiplicities =
            self.table_batch(transcript, proof.trees, (alpha, beta), &sums, &mut table)?;
        // Step 4.
        let (point, evaluations) = self.merge(transcript, &proof.trees.merge, &kept)?;
        self.result(point, evaluations, multiplicities)
    }

    /// Checks 1 to 4.
    fn shapes(&self, proof: Proof<'_>) -> Result<(), Error> {
        count(Message::LogUpRoots, 0, proof.logup.roots.len())?;
        count(Message::LogUpValues, 0, proof.logup.values.len())?;
        count(Message::Groups, self.shapes.len(), proof.trees.groups.len())?;
        for (group, (shape, sent)) in (0..).zip(self.shapes.iter().zip(&proof.trees.groups)) {
            count(Message::Roots { group }, shape.roots, sent.roots.len())?;
            count(
                Message::Levels { group },
                width(shape.levels)?,
                sent.levels.len(),
            )?;
            count(Message::Values { group }, shape.values, sent.values.len())?;
            count(Message::Splits { group }, shape.splits, sent.splits.len())?;
        }
        count(
            Message::TableRoots,
            self.tables.len().checked_mul(2).ok_or(Error::Internal)?,
            proof.trees.table_roots.len(),
        )?;
        count(
            Message::TableLevels,
            width(self.height)?,
            proof.trees.levels.len(),
        )
    }

    fn table_index(&self, table: TableId) -> Result<usize, Error> {
        self.tables
            .binary_search_by_key(&table, |t| t.table)
            .map_err(|_| Error::Internal)
    }

    fn polynomial(&self, index: usize) -> Result<&Polynomial, Error> {
        self.grouped.polynomials().get(index).ok_or(Error::Internal)
    }

    /// Checks 5 to 9 for group `h`, and what it leaves for the merge.
    #[allow(
        clippy::too_many_arguments,
        reason = "one group's step reads the stage's challenges, sums and both public callbacks"
    )]
    fn group(
        &self,
        transcript: &mut impl FiatShamir,
        group: &Group,
        proof: &GroupProof,
        (alpha, beta): (Fp2, Fp2),
        sums: &mut [Option<(Fp2, Fp2)>],
        forest_root: &mut impl FnMut(usize, &[Fp2]) -> Option<Fp2>,
        row_zero: &mut impl FnMut(TableId) -> Option<(Fp, Option<Fp>)>,
    ) -> Result<Kept, Error> {
        let h = group.index;
        let levels_count = group.levels();
        // 5. The roots, `(n, d)` per lookup tree.
        absorb(transcript, &format!("group/{h}/logup/roots"), &proof.roots)?;
        let lookups = group.instances.len();
        let mut entries = vector(
            lookups
                .checked_add(group.forests.len())
                .ok_or(Error::Internal)?,
        )?;
        for (instance, pair) in group.instances.iter().zip(proof.roots.chunks_exact(2)) {
            let &[n, d] = pair else {
                return Err(Error::Internal);
            };
            if d == Fp2::ZERO {
                return Err(Error::Denominator {
                    group: h,
                    table: instance.table,
                });
            }
            let sum = sums
                .get_mut(self.table_index(instance.table)?)
                .ok_or(Error::Internal)?;
            *sum = Some(match *sum {
                Some(earlier) => add(earlier, (n, d)),
                None => (n, d),
            });
            entries.push(Entry::Fraction {
                height: instance.height,
                root: FractionClaim {
                    point: Vec::new(),
                    numerator: n,
                    denominator: d,
                },
            });
        }
        for share in &group.forests {
            entries.push(Entry::Forest {
                height: share.height,
                trees_log: share.trees_log,
            });
        }
        // 6. The levels.
        let leveled = levels(
            transcript,
            |level| Prefix::GroupTree { group: h, level },
            entries,
            &proof.levels,
            |position, at| {
                let share = group.forests.get(position.checked_sub(lookups)?)?;
                forest_root(share.tree, &share.root_point(at)?)
            },
        )
        .map_err(|halt| halt.of(h))?;
        let mut claims = leveled.claims.into_iter();
        // 7. The values and the lookup trees' leaf checks.
        absorb(
            transcript,
            &format!("group/{h}/logup/values"),
            &proof.values,
        )?;
        let mut sent = proof.values.iter();
        let mut leaves = vector(lookups)?;
        for instance in &group.instances {
            let Some(Some(LevelClaim::Fraction(leaf))) = claims.next() else {
                return Err(Error::Internal);
            };
            let value = if instance.has_values {
                Some(*sent.next().ok_or(Error::Internal)?)
            } else {
                None
            };
            if leaf.numerator != Fp2::ONE {
                return Err(Error::LeafNumerator {
                    group: h,
                    table: instance.table,
                });
            }
            let key = alpha - leaf.denominator - beta * value.unwrap_or(Fp2::ZERO);
            leaves.push((leaf.point, key, value));
        }
        // 8. The split and RFC-0045's identity at `P_h`.
        absorb(transcript, &format!("group/{h}/split"), &proof.splits)?;
        let evaluate = |form: &groups::Form| -> Result<Fp2, Error> {
            form.terms
                .iter()
                .try_fold(Fp2::from(form.constant), |sum, &(coefficient, at)| {
                    Ok(sum
                        + Fp2::from(coefficient) * *proof.splits.get(at).ok_or(Error::Internal)?)
                })
        };
        for (instance, (point, key, value)) in group.instances.iter().zip(&leaves) {
            let p = leveled
                .point
                .get(..width(instance.height)?)
                .ok_or(Error::Internal)?;
            // The lookup tree's leaf point is `σ_h` of `P_h[0..h]` (RFC-0045 'The leaf point').
            let leaf_point = p.iter().take(1).chain(p.iter().skip(1).rev());
            if !leaf_point.eq(point.iter()) {
                return Err(Error::Internal);
            }
            let (key0, value0) = row_zero(instance.table).ok_or(Error::Internal)?;
            if value0.is_some() != instance.has_values || value.is_some() != instance.has_values {
                return Err(Error::Internal);
            }
            let padding =
                Fp2::ONE - crate::selectors::lt(p, instance.entries).ok_or(Error::Internal)?;
            let mut expected_key = padding * Fp2::from(key0);
            let mut expected_value = padding * Fp2::from(value0.unwrap_or(Fp::ZERO));
            for block in &instance.blocks {
                let k = width(block.variables)?;
                let mut weight = Fp2::ONE;
                for (bit, &r) in p.get(k..).ok_or(Error::Internal)?.iter().enumerate() {
                    let shift = block
                        .variables
                        .checked_add(u32::try_from(bit).map_err(|_| Error::Internal)?)
                        .ok_or(Error::Internal)?;
                    let high = block.offset.checked_shr(shift).ok_or(Error::Internal)? & 1;
                    weight *= if high == 0 { Fp2::ONE - r } else { r };
                }
                expected_key += weight * evaluate(&block.key)?;
                if let Some(form) = &block.value {
                    expected_value += weight * evaluate(form)?;
                }
            }
            if expected_key != *key || value.is_some_and(|v| v != expected_value) {
                return Err(Error::Split {
                    group: h,
                    level: levels_count,
                });
            }
        }
        // 9. γ_h, then the group's share of the merge's claim.
        let gamma = challenge(transcript, &format!("group/{h}/merge/gamma"))?;
        let mut forests = vector(group.forests.len())?;
        for _ in &group.forests {
            let Some(Some(LevelClaim::Product(claim))) = claims.next() else {
                return Err(Error::Internal);
            };
            forests.push(claim);
        }
        if claims.next().is_some() {
            return Err(Error::Internal);
        }
        let mut claimed = Fp2::ZERO;
        let mut power = Fp2::ONE;
        let values = group
            .split
            .iter()
            .zip(&proof.splits)
            .map(|(v, &x)| (v.polynomial, x));
        let shares = group
            .forests
            .iter()
            .zip(&forests)
            .map(|(s, c)| (s.polynomial, c.value));
        for (polynomial, value) in values.chain(shares) {
            let n = self.polynomial(polynomial)?.variables;
            let padding = two_to(self.n_max.checked_sub(n).ok_or(Error::Internal)?);
            claimed += power * Fp2::from(padding) * value;
            power *= gamma;
        }
        Ok(Kept {
            gamma,
            point: leveled.point,
            forests,
            claimed,
        })
    }

    /// Checks 10 to 13, and the multiplicity claims.
    fn table_batch(
        &self,
        transcript: &mut impl FiatShamir,
        trees: &TreeProof,
        (alpha, beta): (Fp2, Fp2),
        sums: &[Option<(Fp2, Fp2)>],
        table: &mut impl FnMut(TableId, Fp2, &[Fp2]) -> Option<Fp2>,
    ) -> Result<Vec<(TableId, Claim)>, Error> {
        // 10.
        absorb(transcript, "tables/logup/roots", &trees.table_roots)?;
        let roots = trees.table_roots.chunks_exact(2);
        for (tree, pair) in self.tables.iter().zip(roots.clone()) {
            if pair.get(1) == Some(&Fp2::ZERO) {
                return Err(Error::TableDenominator { table: tree.table });
            }
        }
        // 11.
        let mut entries = vector(self.tables.len())?;
        for ((tree, pair), sum) in self.tables.iter().zip(roots).zip(sums) {
            let &[n, d] = pair else {
                return Err(Error::Internal);
            };
            let (numerator, _) = add(sum.ok_or(Error::Internal)?, (n, d));
            if numerator != Fp2::ZERO {
                return Err(Error::Unbalanced { table: tree.table });
            }
            entries.push(Entry::Fraction {
                height: tree.height,
                root: FractionClaim {
                    point: Vec::new(),
                    numerator: n,
                    denominator: d,
                },
            });
        }
        // 12.
        let groups = self.groups();
        let leveled = levels(
            transcript,
            Prefix::TableTree,
            entries,
            &trees.levels,
            |_, _| None,
        )
        .map_err(|halt| halt.of(groups))?;
        // 13.
        let mut out = vector(self.tables.len())?;
        for (tree, claim) in self.tables.iter().zip(leveled.claims) {
            let Some(LevelClaim::Fraction(leaf)) = claim else {
                return Err(Error::Internal);
            };
            let evaluation = table(tree.table, beta, &leaf.point).ok_or(Error::Internal)?;
            if leaf.denominator != alpha - evaluation {
                return Err(Error::TableEvaluation { table: tree.table });
            }
            out.push((
                tree.table,
                Claim {
                    point: leaf.point,
                    value: -leaf.numerator,
                },
            ));
        }
        Ok(out)
    }

    /// Checks 14 to 17: the merge's point and its evaluations.
    fn merge(
        &self,
        transcript: &mut impl FiatShamir,
        proof: &SumcheckProof,
        kept: &[Kept],
    ) -> Result<(Vec<Fp2>, Vec<Fp2>), Error> {
        // 14.
        let beta = challenge(transcript, "merge/beta")?;
        // 15.
        let mut claim = Fp2::ZERO;
        let mut power = Fp2::ONE;
        for group in kept {
            claim += power * group.claimed;
            power *= beta;
        }
        let reduction = sumcheck::reduce_with(
            transcript,
            Prefix::Merge,
            MERGE_DEGREE,
            width(self.n_max)?,
            claim,
            &proof.rounds,
        )
        .map_err(Error::Merge)?;
        // 16.
        let polynomials = self.grouped.polynomials();
        let round = self.n_max;
        if proof.final_evals.len() != polynomials.len() {
            return Err(Error::Merge(SumcheckError::EvaluationCount {
                round,
                expected: polynomials.len(),
                found: proof.final_evals.len(),
            }));
        }
        fiat_shamir::absorb(transcript, Prefix::Merge, Step::Evals, &proof.final_evals)
            .map_err(Error::Merge)?;
        // 17. Every `W_i(r_i)`, group by group, each group's layout derived once more.
        let mut weights = vector(polynomials.len())?;
        weights.resize(polynomials.len(), Fp2::ZERO);
        let mut claims: Vec<u32> = vector(polynomials.len())?;
        claims.resize(polynomials.len(), 0);
        let mut power = Fp2::ONE;
        for (h, group) in (0..self.groups()).zip(kept) {
            let layout = self.grouped.group(h).map_err(layout)?;
            let mut gamma_k = Fp2::ONE;
            let mut add_claim = |index: usize, free: Outside<'_>, bits: u32| -> Result<(), Error> {
                let polynomial = self.polynomial(index)?;
                if !polynomial.readers.contains(&h) {
                    return Err(Error::Internal);
                }
                let own = suffix(&reduction.point, polynomial.variables)?;
                let fixed = polynomial.group_variables.len();
                let (outside, inside) = own
                    .split_at_checked(own.len().checked_sub(fixed).ok_or(Error::Internal)?)
                    .ok_or(Error::Internal)?;
                let factor = free.eq(outside)? * eq_bits(bits, inside)?;
                *weights.get_mut(index).ok_or(Error::Internal)? += power * gamma_k * factor;
                let seen = claims.get_mut(index).ok_or(Error::Internal)?;
                *seen = seen.checked_add(1).ok_or(Error::Internal)?;
                gamma_k *= group.gamma;
                Ok(())
            };
            for value in &layout.split {
                add_claim(
                    value.polynomial,
                    Outside::Coordinates(&group.point, &value.coordinates),
                    value.bits,
                )?;
            }
            if layout.forests.len() != group.forests.len() {
                return Err(Error::Internal);
            }
            for (share, claim) in layout.forests.iter().zip(&group.forests) {
                add_claim(share.polynomial, Outside::Point(&claim.point), share.bits)?;
            }
            power *= beta;
        }
        for (polynomial, &seen) in polynomials.iter().zip(&claims) {
            let readers = polynomial
                .readers
                .end
                .checked_sub(polynomial.readers.start)
                .ok_or(Error::Internal)?;
            if seen == 0 || seen != readers {
                return Err(Error::Internal);
            }
        }
        let value = weights
            .iter()
            .zip(&proof.final_evals)
            .fold(Fp2::ZERO, |sum, (&w, &e)| sum + w * e);
        if value != reduction.claim {
            return Err(Error::Merge(SumcheckError::FinalMismatch { round }));
        }
        Ok((reduction.point, copied(&proof.final_evals).map_err(own)?))
    }

    /// Each merged polynomial's claim at its own point, in its natural variable order.
    fn result(
        &self,
        point: Vec<Fp2>,
        evaluations: Vec<Fp2>,
        multiplicities: Vec<(TableId, Claim)>,
    ) -> Result<Merged, Error> {
        let polynomials = self.grouped.polynomials();
        let mut columns = vector(self.grouped.values())?;
        let mut forests = vector(polynomials.len().saturating_sub(self.grouped.values()))?;
        for (polynomial, value) in polynomials.iter().zip(evaluations) {
            let own = suffix(&point, polynomial.variables)?;
            let n = width(polynomial.variables)?;
            let fixed = &polynomial.group_variables;
            let free = n.checked_sub(fixed.len()).ok_or(Error::Internal)?;
            let mut natural = vector(n)?;
            let (mut outside, mut inside) = (0usize, 0usize);
            for variable in 0..polynomial.variables {
                let slot = if fixed.get(inside) == Some(&variable) {
                    inside = inside.checked_add(1).ok_or(Error::Internal)?;
                    free.checked_add(inside).and_then(|s| s.checked_sub(1))
                } else {
                    outside = outside.checked_add(1).ok_or(Error::Internal)?;
                    outside.checked_sub(1).filter(|&s| s < free)
                };
                natural.push(
                    *own.get(slot.ok_or(Error::Internal)?)
                        .ok_or(Error::Internal)?,
                );
            }
            if inside != fixed.len() {
                return Err(Error::Internal);
            }
            match polynomial.source {
                Source::Split { node, column } => columns.push(ColumnClaim {
                    node,
                    column,
                    point: natural,
                    value,
                }),
                Source::Forest { tree, .. } => forests.push(ForestClaim {
                    tree,
                    claim: TreeClaim {
                        point: natural,
                        value,
                    },
                }),
            }
        }
        Ok(Merged {
            columns,
            forests,
            multiplicities,
            point,
        })
    }
}

/// A claim's point outside its group variables, as `W_i` reads it: a split value's coordinates in
/// `P_h`, or a share's leaf point.
enum Outside<'p> {
    Coordinates(&'p [Fp2], &'p [u32]),
    Point(&'p [Fp2]),
}
impl Outside<'_> {
    /// `eq(z, r)`.
    fn eq(&self, r: &[Fp2]) -> Result<Fp2, Error> {
        match *self {
            Self::Coordinates(point, coordinates) => {
                if coordinates.len() != r.len() {
                    return Err(Error::Internal);
                }
                let mut out = Fp2::ONE;
                for (&c, &x) in coordinates.iter().zip(r) {
                    out *= eq1(*point.get(width(c)?).ok_or(Error::Internal)?, x);
                }
                Ok(out)
            }
            Self::Point(z) => sumcheck::eq(z, r).ok_or(Error::Internal),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::stub::{Op, Stub};
    use crate::fiat_shamir::tce;
    use crate::leaf::groups::Extent;
    use crate::leaf::split::{Affine, Column, Layer, Lookup, Read, Site, TokenAxis};
    use crate::lowering::NodeId;
    use crate::protocol::{GroupProof, LeafGroups};

    /// One site of one Range lookup over `[4, 2]`, its token axis on variables 1 and 2, in two
    /// token groups.
    fn grouped() -> Grouped {
        let site = Site {
            layers: vec![Layer {
                node: NodeId(3),
                lookups: vec![Lookup {
                    table: TableId(1),
                    variables: 3,
                    key: Affine {
                        constant: Fp::ZERO,
                        terms: vec![(
                            Fp::ONE,
                            Read {
                                column: Column::Input(0),
                                coordinates: vec![0, 1, 2],
                            },
                        )],
                    },
                    value: None,
                    token: Some(TokenAxis { low: 1, bits: 2 }),
                }],
            }],
        };
        let extent = Extent {
            token_bits: 2,
            layers: 1,
        };
        let groups = LeafGroups {
            token_bits: 1,
            layer_bits: 0,
        };
        Grouped::new(&[site], &[], extent, groups).unwrap()
    }
    const TABLES: [TableTree; 1] = [TableTree {
        table: TableId(1),
        height: 2,
    }];
    /// Messages of the right counts, every value 1.
    fn messages(stage: &Stage<'_>) -> (LogUpProof, TreeProof) {
        let groups = stage
            .shapes
            .iter()
            .map(|shape| GroupProof {
                roots: vec![Fp2::ONE; shape.roots],
                levels: vec![SumcheckProof::default(); shape.levels as usize],
                values: vec![Fp2::ONE; shape.values],
                splits: vec![Fp2::ONE; shape.splits],
            })
            .collect();
        (
            LogUpProof {
                roots: vec![],
                values: vec![],
            },
            TreeProof {
                levels: vec![SumcheckProof::default(); stage.height as usize],
                groups,
                table_roots: vec![Fp2::ONE; 2],
                merge: SumcheckProof::default(),
                splits: vec![],
            },
        )
    }

    /// RFC-0014 row 4 with groups, then group 0's roots: `α`, `β`, the two empty messages, and
    /// the roots, whose zero denominator stops the check before any level.
    #[test]
    fn the_challenges_and_the_empty_messages_precede_group_zeros_roots() {
        let grouped = grouped();
        let stage = Stage::new(&grouped, &TABLES).unwrap();
        assert_eq!(
            (stage.groups(), stage.table_levels(), stage.rounds()),
            (2, 2, 3)
        );
        let (logup, mut trees) = messages(&stage);
        trees.groups[0].roots[1] = Fp2::ZERO;
        let mut stub = Stub::default();
        let outcome = stage.verify_with(
            &mut stub,
            Proof {
                logup: &logup,
                trees: &trees,
            },
            |_, _| None,
            |_, _, _| None,
            |_| None,
        );
        assert_eq!(
            outcome,
            Err(Error::Denominator {
                group: 0,
                table: TableId(1)
            })
        );
        assert_eq!(
            stub.script(),
            [
                "?logup/alpha",
                "?logup/beta",
                "+logup/roots",
                "+logup/values",
                "+group/0/logup/roots"
            ]
        );
        assert_eq!(stub.ops[2], Op::Absorb("logup/roots".into(), tce(&[])));
        // A message of another count is judged before the transcript moves.
        trees.table_roots.push(Fp2::ONE);
        let mut stub = Stub::default();
        let outcome = stage.verify_with(
            &mut stub,
            Proof {
                logup: &logup,
                trees: &trees,
            },
            |_, _| None,
            |_, _, _| None,
            |_| None,
        );
        assert_eq!(
            outcome,
            Err(Error::Count {
                message: Message::TableRoots,
                expected: 2,
                found: 3
            })
        );
        assert!(stub.ops.is_empty());
    }

    /// A challenge without value, `α`, `β` or a level's, is a fault of the verifier.
    #[test]
    fn an_exhausted_challenge_is_a_fault_of_the_verifier() {
        let grouped = grouped();
        let stage = Stage::new(&grouped, &TABLES).unwrap();
        let (logup, trees) = messages(&stage);
        let exhausted = TranscriptError::Exhausted;
        for (fail_at, expected) in [
            (0, Error::Transcript(exhausted)),
            (1, Error::Transcript(exhausted)),
            (
                2,
                Error::Level {
                    group: 0,
                    level: 0,
                    error: SumcheckError::Transcript(exhausted),
                },
            ),
        ] {
            let mut stub = Stub::failing_at(fail_at);
            let outcome = stage.verify_with(
                &mut stub,
                Proof {
                    logup: &logup,
                    trees: &trees,
                },
                |_, _| None,
                |_, _, _| None,
                |_| None,
            );
            assert_eq!(outcome, Err(expected));
            assert_eq!(expected.rejection(ArtifactKind(0x0520)), None);
            assert!(matches!(
                expected.verdict(ArtifactKind(0x0520)),
                Verdict::Inconclusive(VerifierError {
                    code: ErrorCode::Internal
                })
            ));
        }
    }

    /// RFC-0014's label patterns for a grouped stage.
    #[test]
    fn the_grouped_prefixes_spell_rfc_0014s_labels() {
        let group = Prefix::GroupTree { group: 3, level: 2 };
        assert_eq!(group.to_string(), "group/3/tree/2");
        assert_eq!(Prefix::TableTree(1).to_string(), "tables/tree/1");
        assert_eq!(Prefix::Merge.to_string(), "merge");
        let mut stub = Stub::default();
        let _ = fiat_shamir::challenge(&mut stub, group, Step::Root(4));
        let _ = fiat_shamir::absorb(&mut stub, Prefix::Merge, Step::Round(0), &[]);
        let _ = fiat_shamir::challenge(&mut stub, Prefix::Merge, Step::Challenge(0));
        let _ = fiat_shamir::absorb(&mut stub, Prefix::Merge, Step::Evals, &[]);
        assert_eq!(
            stub.script(),
            [
                "?group/3/tree/2/root/4",
                "+merge/round/0",
                "?merge/r/0",
                "+merge/evals"
            ]
        );
    }

    /// Each error's verdict (04 §2, RFC-0054 'Messages').
    #[test]
    fn each_error_maps_to_its_reason() {
        let artifact = ArtifactKind(0x0511);
        let table = TableId(7);
        let cases = [
            (
                Error::Count {
                    message: Message::Groups,
                    expected: 2,
                    found: 1,
                },
                Some(RejectReason::Malformed {
                    artifact,
                    code: ErrorCode::CodecNoncanonical,
                }),
            ),
            (
                Error::Denominator { group: 1, table },
                Some(RejectReason::Lookup { table }),
            ),
            (
                Error::LeafNumerator { group: 1, table },
                Some(RejectReason::Lookup { table }),
            ),
            (
                Error::TableDenominator { table },
                Some(RejectReason::Lookup { table }),
            ),
            (
                Error::Unbalanced { table },
                Some(RejectReason::Lookup { table }),
            ),
            (
                Error::TableEvaluation { table },
                Some(RejectReason::Lookup { table }),
            ),
            (
                Error::Level {
                    group: 4,
                    level: 2,
                    error: SumcheckError::RoundLength {
                        round: 1,
                        expected: 3,
                        found: 2,
                    },
                },
                Some(RejectReason::TreeLevel {
                    group: 4,
                    level: 2,
                    round: 1,
                }),
            ),
            (
                Error::Split { group: 3, level: 5 },
                Some(RejectReason::TreeLevel {
                    group: 3,
                    level: 5,
                    round: 0,
                }),
            ),
            (
                Error::Merge(SumcheckError::FinalMismatch { round: 9 }),
                Some(RejectReason::LeafMerge { round: 9 }),
            ),
            (Error::Merge(SumcheckError::NoInstances), None),
            (Error::Internal, None),
        ];
        for (error, reason) in cases {
            assert_eq!(error.rejection(artifact), reason, "{error}");
        }
        for (error, code) in [
            (Error::Memory, ErrorCode::Memory),
            (
                Error::Transcript(TranscriptError::Memory),
                ErrorCode::Memory,
            ),
            (
                Error::Merge(SumcheckError::Transcript(TranscriptError::Memory)),
                ErrorCode::Memory,
            ),
            (Error::Internal, ErrorCode::Internal),
            (
                Error::Level {
                    group: 0,
                    level: 0,
                    error: SumcheckError::Relation { instance: 0 },
                },
                ErrorCode::Internal,
            ),
        ] {
            assert!(
                matches!(error.verdict(artifact), Verdict::Inconclusive(e) if e.code == code),
                "{error}"
            );
        }
    }
}
