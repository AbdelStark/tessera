// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0045's lookup-column split, after the shared tree checks and before any stage challenge.
//! Metadata is verifier-owned. The result retains every source, multiplicity and forest debt;
//! checking an affine lookup identity is not verification of the tensor values themselves.

use super::{ForestClaim, Verified, copied, reserved};
use crate::advice::Auxiliary;
use crate::fiat_shamir::{self, FiatShamir};
use crate::logup::Claim;
use crate::lowering::NodeId;
mod model;
pub use model::model_sites;
pub(crate) use model::sites as model_sites_with;
pub(crate) use model::token as token_axis;

use core::fmt;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::Transcript;
use tv_tables::TableId;

/// A column local to one gadget node. Virtual columns allocate no commitment positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Column {
    /// The original graph input at this node's input index.
    Input(u8),
    /// The graph output at this node's output index.
    Output(u8),
    /// A committed RFC-0050 role at this node.
    Auxiliary(Auxiliary),
    /// Rsqrt's virtual full input `u`; its original residual/sum-of-squares binding is still owed.
    RootInput,
    /// Softmax's virtual masked Exp key `kappa`, reduced with the public causal selector.
    SoftmaxKey,
    /// Softmax's virtual first bracket slack `a`.
    SoftmaxA,
    /// Softmax's virtual second bracket slack `b`.
    SoftmaxB,
    /// Runtime greedy Range key, owed to the native original-source reduction.
    GreedyKey,
}

/// One column read at public coordinates of RFC-0045's common point `P`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Read {
    /// Its local identity.
    pub column: Column,
    /// Its natural LSB-first point takes these coordinates of `P`, in this order.
    pub coordinates: Vec<u32>,
}

/// An affine lookup key/value form, terms in the normative formula's first-read order.
#[derive(Clone, Debug)]
pub struct Affine {
    /// Public constant coefficient.
    pub constant: Fp,
    /// Coefficient and column read; duplicate reads add but send only one value.
    pub terms: Vec<(Fp, Read)>,
}

/// Where a block's token axis lies among its variables (RFC-0054 'The token axis'): the domain's
/// axes after it take the lowest variables (RFC-0045 'Blocks', index order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenAxis {
    /// The block variable that holds the token index's lowest bit.
    pub low: u32,
    /// The token axis's variables, `⌈log2 n⌉` for `n` tokens.
    pub bits: u32,
}

/// One canonical lookup request over its padded domain; repeated tables remain distinct.
#[derive(Clone, Debug)]
pub struct Lookup {
    /// The verifier-derived table.
    pub table: TableId,
    /// Sum of the domain axes' ceil-log2 lengths.
    pub variables: u32,
    /// Key form.
    pub key: Affine,
    /// Value form; `None` exactly for a Range table.
    pub value: Option<Affine>,
    /// The domain's token axis, its lookup tensor's (RFC-0054 'The token axis'); `None` for a
    /// domain without one, whose block a grouped stage holds whole in token subset 0.
    /// [`groups`](super::groups) lays the groups out by it, and [`LeafPoint::read`] places a
    /// grouped stage's claims by it; an ungrouped stage's points ignore it (RFC-0045).
    pub token: Option<TokenAxis>,
}

/// One site's real layer, or its only layer outside the template.
#[derive(Clone, Debug)]
pub struct Layer {
    /// The original graph node, never a proof-supplied identifier.
    pub node: NodeId,
    /// Requests in RFC-0009 order.
    pub lookups: Vec<Lookup>,
}

/// A verifier-owned site, in first-node order, with one entry per real layer.
#[derive(Clone, Debug)]
pub struct Site {
    /// Real layers in order; padded layers have no lookup block or split value.
    pub layers: Vec<Layer>,
}

#[derive(Debug)]
struct Form {
    constant: Fp,
    terms: Vec<(Fp, usize)>,
}
#[derive(Debug)]
struct Block {
    layer: usize,
    variables: u32,
    offset: u64,
    order: usize,
    key: Form,
    value: Option<Form>,
}
#[derive(Debug)]
struct Instance {
    table: TableId,
    entries: u64,
    height: u32,
    blocks: Vec<Block>,
    has_values: bool,
}
#[derive(Debug)]
struct Inputs {
    node: NodeId,
    reads: Vec<Read>,
    /// Each read's block token axis, its first reader's (RFC-0054 'Agreement').
    tokens: Vec<Option<TokenAxis>>,
    start: usize,
}

/// The independently derived blocks, offsets and first-read ordering of a leaf split.
#[derive(Debug)]
pub struct Layout {
    inputs: Vec<Inputs>,
    instances: Vec<Instance>,
    values: usize,
}

/// A source evaluation still owed to a graph/native reduction or canonical opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnClaim {
    /// Its actual graph node, including its real layer.
    pub node: NodeId,
    /// Its local column identity; virtual variants must be reduced, not opened as advice.
    pub column: Column,
    /// Its prescribed natural point, derived from `P`.
    pub point: Vec<Fp2>,
    /// The fixed value from the split message.
    pub value: Fp2,
}

/// Where the leaf stage leaves its claims on tensors: the one rule by which the gadget
/// reductions, the stages and the openings place every claim the stage leaves (RFC-0045 'The
/// leaf point'; RFC-0054 'The stage's outputs').
///
/// A claim lies on a tensor of `n` variables, in its natural (LSB-first) variable order. Its
/// group variables `S` are the low `g_T` variables of the tensor's token axis (the axes after it
/// take the lower variables, RFC-0045 'Blocks'); a tensor without a token axis has none.
///
/// - [`Shared`](Self::Shared), an ungrouped stage: the claim lies at `P[0..n]`, and a read at
///   its coordinates of `P`.
/// - [`Merged`](Self::Merged), a grouped stage: it lies at `r(n, S)`, the merge's last `n`
///   challenges assigned to the variables outside `S`, lowest first, then to those in `S`,
///   lowest first (RFC-0054 'The merged point').
///
/// A site's tensors of one shape and one token axis therefore share their point in both, as
/// RFC-0037's layer point requires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeafPoint {
    /// RFC-0045's `P`: the last level's line challenge, then its round challenges from the last.
    /// It is empty when no tree has a level.
    Shared(Vec<Fp2>),
    /// The merge's point (RFC-0054 'The merge').
    Merged {
        /// The merge's challenges, in round order: `n_max` of them.
        challenges: Vec<Fp2>,
        /// `g_T`, how many low variables of a token axis a group fixes.
        token_bits: u32,
    },
}

impl LeafPoint {
    /// The point of the stage's claim on a tensor of `variables` variables whose token axis is
    /// `token` (module documentation of [`LeafPoint`]). `None` when the rule has no such point:
    /// more variables than the stage's point holds, or a token axis outside the tensor.
    pub fn tensor(&self, variables: u32, token: Option<TokenAxis>) -> Option<Vec<Fp2>> {
        let n = usize::try_from(variables).ok()?;
        match self {
            Self::Shared(p) => {
                let mut out = Vec::new();
                out.try_reserve_exact(n).ok()?;
                out.extend_from_slice(p.get(..n)?);
                Some(out)
            }
            Self::Merged {
                challenges,
                token_bits,
            } => {
                let mut fixed = Vec::new();
                if let Some(axis) = token {
                    let end = axis.low.checked_add(*token_bits)?;
                    if *token_bits > axis.bits || axis.low.checked_add(axis.bits)? > variables {
                        return None;
                    }
                    fixed
                        .try_reserve_exact(usize::try_from(*token_bits).ok()?)
                        .ok()?;
                    fixed.extend(axis.low..end);
                }
                merged(challenges, variables, &fixed)
            }
        }
    }

    /// The point of a read at `coordinates` of a block whose domain's token axis is `token`: in
    /// an ungrouped stage `P` at those coordinates; in a grouped one `r(m, S)` for its `m`
    /// variables, `S` those whose coordinate lies among the token axis's low `g_T` (RFC-0054
    /// 'Group variables'). `None` when the rule has no such point.
    pub fn read(&self, coordinates: &[u32], token: Option<TokenAxis>) -> Option<Vec<Fp2>> {
        match self {
            Self::Shared(p) => {
                let mut out = Vec::new();
                out.try_reserve_exact(coordinates.len()).ok()?;
                for &c in coordinates {
                    out.push(*p.get(usize::try_from(c).ok()?)?);
                }
                Some(out)
            }
            Self::Merged {
                challenges,
                token_bits,
            } => {
                let mut fixed = Vec::new();
                if let Some(axis) = token {
                    let cut = axis.low.checked_add(*token_bits)?;
                    for (variable, &c) in (0u32..).zip(coordinates) {
                        if c >= axis.low && c < cut {
                            fixed.try_reserve(1).ok()?;
                            fixed.push(variable);
                        }
                    }
                }
                merged(challenges, u32::try_from(coordinates.len()).ok()?, &fixed)
            }
        }
    }
}

/// `r(n, S)` in natural variable order: the last `n` of `challenges`, the first `n − |S|` to the
/// variables outside `S` lowest first, the rest to those in `S` (ascending) lowest first.
fn merged(challenges: &[Fp2], variables: u32, fixed: &[u32]) -> Option<Vec<Fp2>> {
    let n = usize::try_from(variables).ok()?;
    let own = challenges.get(challenges.len().checked_sub(n)?..)?;
    let free = n.checked_sub(fixed.len())?;
    let mut out = Vec::new();
    out.try_reserve_exact(n).ok()?;
    let (mut outside, mut inside) = (0usize, 0usize);
    for variable in 0..variables {
        let slot = if fixed.get(inside) == Some(&variable) {
            inside = inside.checked_add(1)?;
            free.checked_add(inside)?.checked_sub(1)?
        } else {
            outside = outside.checked_add(1)?;
            outside.checked_sub(1).filter(|&s| s < free)?
        };
        out.push(*own.get(slot)?);
    }
    (inside == fixed.len()).then_some(out)
}

/// The debts after checking the lookup split; not an acceptance verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "check all source/forest reductions and multiplicity openings, and the graph atom splits"]
pub struct Bound {
    /// Every distinct source, in site/layer/first-read order, each at its point by
    /// [`LeafPoint::read`].
    pub columns: Vec<ColumnClaim>,
    /// Table multiplicities retain their original table-tree points and table identities.
    pub multiplicities: Vec<(TableId, Claim)>,
    /// All original product-forest leaf claims, at their own points.
    pub forests: Vec<ForestClaim>,
    /// Graph atom split values: the tail of the same absorbed vector in an ungrouped stage, the
    /// whole of `stage/0/split` in a grouped one (RFC-0054 step 5).
    pub atoms: Vec<Fp2>,
    /// Where the stage left its claims.
    pub point: LeafPoint,
}

/// The first failed split check or a verifier fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// `Reject(Malformed)`: lookup values plus the public atom count do not fill the message.
    Count {
        /// Expected public count.
        expected: usize,
        /// Message count.
        found: usize,
    },
    /// `Reject(Sumcheck { stage: 0, round: 0 })`: a key or value identity failed.
    Mismatch,
    /// `Inconclusive(ERR-RES-MEMORY)`.
    Memory,
    /// `Inconclusive(ERR-INTERNAL)`: inconsistent own layout/points or absent public row 0.
    Internal,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count { expected, found } => write!(
                f,
                "the leaf split holds {found} values, expected {expected}"
            ),
            Self::Mismatch => f.write_str("the leaf lookup split does not match its column claim"),
            Self::Memory => f.write_str("memory for leaf split checks could not be reserved"),
            Self::Internal => f.write_str("inconsistent verifier-owned lookup split metadata"),
        }
    }
}
impl std::error::Error for Error {}
fn own(error: super::Error) -> Error {
    match error {
        super::Error::Memory => Error::Memory,
        _ => Error::Internal,
    }
}
fn vector<T>(n: usize) -> Result<Vec<T>, Error> {
    reserved(n).map_err(own)
}

impl Layout {
    /// Compile public canonical forms into first-read slots and aligned table blocks.
    /// This constructor is for verifier-derived descriptors, never decoded proof metadata.
    /// The caller's resource policy must bound the public site/request count.
    ///
    /// # Errors
    /// `Internal` for unordered/empty sites, inconsistent real layers or column coordinates,
    /// conflicting value-table kinds, and dimensions outside 63 variables; `Memory` on reserve.
    pub fn new(sites: &[Site]) -> Result<Self, Error> {
        let mut inputs = Vec::new();
        let mut instances: Vec<Instance> = Vec::new();
        let mut count = 0usize;
        let mut previous = None;
        for site in sites {
            let first = site.layers.first().ok_or(Error::Internal)?.node;
            if previous.is_some_and(|id: NodeId| id.0 >= first.0) {
                return Err(Error::Internal);
            }
            previous = Some(first);
            let mut first_reads = None;
            let mut last_node = None;
            for layer in &site.layers {
                if !site
                    .layers
                    .first()
                    .ok_or(Error::Internal)?
                    .lookups
                    .iter()
                    .map(|l| l.variables)
                    .eq(layer.lookups.iter().map(|l| l.variables))
                {
                    return Err(Error::Internal);
                }
                if last_node.is_some_and(|id: NodeId| id.0 >= layer.node.0) {
                    return Err(Error::Internal);
                }
                last_node = Some(layer.node);
                let mut reads = Vec::new();
                let mut tokens: Vec<Option<TokenAxis>> = Vec::new();
                for lookup in &layer.lookups {
                    if lookup.variables > 63 {
                        return Err(Error::Internal);
                    }
                    let mut form = |source: &Affine| -> Result<Form, Error> {
                        let mut terms = vector(source.terms.len())?;
                        for &(coefficient, ref read) in &source.terms {
                            if read.coordinates.iter().enumerate().any(|(i, &c)| {
                                c >= lookup.variables
                                    || read
                                        .coordinates
                                        .get(..i)
                                        .is_some_and(|earlier| earlier.contains(&c))
                            }) {
                                return Err(Error::Internal);
                            }
                            let index =
                                match reads.iter().position(|r: &Read| r.column == read.column) {
                                    Some(index) => {
                                        if reads.get(index) != Some(read) {
                                            return Err(Error::Internal);
                                        }
                                        index
                                    }
                                    None => {
                                        let index = reads.len();
                                        reads.try_reserve(1).map_err(|_| Error::Memory)?;
                                        tokens.try_reserve(1).map_err(|_| Error::Memory)?;
                                        reads.push(Read {
                                            column: read.column,
                                            coordinates: copied(&read.coordinates).map_err(own)?,
                                        });
                                        tokens.push(lookup.token);
                                        index
                                    }
                                };
                            terms.push((coefficient, index));
                        }
                        Ok(Form {
                            constant: source.constant,
                            terms,
                        })
                    };
                    let key = form(&lookup.key)?;
                    let value = lookup.value.as_ref().map(&mut form).transpose()?;
                    let position = match instances.iter().position(|i| i.table == lookup.table) {
                        Some(index) => index,
                        None => {
                            instances.try_reserve(1).map_err(|_| Error::Memory)?;
                            instances.push(Instance {
                                table: lookup.table,
                                entries: 0,
                                height: 0,
                                blocks: Vec::new(),
                                has_values: value.is_some(),
                            });
                            instances.len().checked_sub(1).ok_or(Error::Internal)?
                        }
                    };
                    let instance = instances.get_mut(position).ok_or(Error::Internal)?;
                    if instance.has_values != value.is_some() {
                        return Err(Error::Internal);
                    }
                    instance.blocks.try_reserve(1).map_err(|_| Error::Memory)?;
                    let order = instance.blocks.len();
                    instance.blocks.push(Block {
                        layer: inputs.len(),
                        order,
                        variables: lookup.variables,
                        offset: 0,
                        key,
                        value,
                    });
                }
                if let Some(first) = first_reads {
                    if inputs.get(first).map(|i: &Inputs| &i.reads) != Some(&reads) {
                        return Err(Error::Internal);
                    }
                } else {
                    first_reads = Some(inputs.len());
                }
                inputs.try_reserve(1).map_err(|_| Error::Memory)?;
                let start = count;
                count = count.checked_add(reads.len()).ok_or(Error::Internal)?;
                inputs.push(Inputs {
                    node: layer.node,
                    reads,
                    tokens,
                    start,
                });
            }
        }
        instances.sort_unstable_by_key(|i| i.table);
        for instance in &mut instances {
            // Explicit ordinals preserve RFC-0045's stable ties without a sorting allocation.
            instance
                .blocks
                .sort_unstable_by_key(|b| (core::cmp::Reverse(b.variables), b.order));
            for block in &mut instance.blocks {
                block.offset = instance.entries;
                instance.entries = instance
                    .entries
                    .checked_add(1u64.checked_shl(block.variables).ok_or(Error::Internal)?)
                    .ok_or(Error::Internal)?;
            }
            instance.height =
                u64::BITS.saturating_sub(instance.entries.saturating_sub(1).leading_zeros());
            if instance.entries == 0 || instance.height > 63 {
                return Err(Error::Internal);
            }
        }
        Ok(Self {
            inputs,
            instances,
            values: count,
        })
    }

    /// Number of lookup tensor values, before the graph atom tail.
    pub fn values(&self) -> usize {
        self.values
    }

    pub(crate) fn check_bound(&self, bound: &Bound) -> Result<(), Error> {
        if bound.columns.len() != self.values {
            return Err(Error::Internal);
        }
        for input in &self.inputs {
            for (i, (read, &token)) in input.reads.iter().zip(&input.tokens).enumerate() {
                let at = input.start.checked_add(i).ok_or(Error::Internal)?;
                let column = bound.columns.get(at).ok_or(Error::Internal)?;
                if column.node != input.node
                    || column.column != read.column
                    || bound.point.read(&read.coordinates, token).as_ref() != Some(&column.point)
                {
                    return Err(Error::Internal);
                }
            }
        }
        Ok(())
    }

    /// Used tables and their unrounded column lengths, in ascending table order.
    pub fn instances(&self) -> impl Iterator<Item = (TableId, u64)> + '_ {
        self.instances.iter().map(|i| (i.table, i.entries))
    }

    /// Absorb the complete split once and check every RFC-0045 column identity.
    /// `row_zero` supplies the verifier's canonical public row 0, including whether it has a
    /// value. `atom_count` is from the public graph/native schedule, never the proof.
    /// No challenge is drawn. Source, multiplicity, forest and atom obligations are retained.
    ///
    /// # Errors
    /// `Count` before transcript work; then own point/row metadata faults, or `Mismatch`
    /// at the first table whose key/value does not match. All new allocations are bounded
    /// by public counts and reserve fallibly.
    pub fn bind(
        &self,
        transcript: &mut Transcript,
        leaf: Verified,
        splits: &[Fp2],
        atom_count: usize,
        row_zero: impl FnMut(TableId) -> Option<(Fp, Option<Fp>)>,
    ) -> Result<Bound, Error> {
        self.bind_with(transcript, leaf, splits, atom_count, row_zero)
    }

    fn bind_with(
        &self,
        transcript: &mut impl FiatShamir,
        leaf: Verified,
        splits: &[Fp2],
        atom_count: usize,
        mut row_zero: impl FnMut(TableId) -> Option<(Fp, Option<Fp>)>,
    ) -> Result<Bound, Error> {
        let expected = self.values.checked_add(atom_count).ok_or(Error::Internal)?;
        if splits.len() != expected {
            return Err(Error::Count {
                expected,
                found: splits.len(),
            });
        }
        if leaf.lookups.instances.len() != self.instances.len() {
            return Err(Error::Internal);
        }
        fiat_shamir::absorb_under(transcript, "stage/0/split", splits)
            .map_err(|_| Error::Internal)?;
        let evaluate = |form: &Form, layer: usize| -> Result<Fp2, Error> {
            let input = self.inputs.get(layer).ok_or(Error::Internal)?;
            form.terms
                .iter()
                .try_fold(Fp2::from(form.constant), |sum, &(coefficient, index)| {
                    let at = input.start.checked_add(index).ok_or(Error::Internal)?;
                    Ok(sum
                        + Fp2::from(coefficient)
                            * splits.get(at).copied().ok_or(Error::Internal)?)
                })
        };
        for (instance, claims) in self.instances.iter().zip(&leaf.lookups.instances) {
            let h = usize::try_from(instance.height).map_err(|_| Error::Internal)?;
            let p = leaf.point.get(..h).ok_or(Error::Internal)?;
            let mut expected_point = vector(h)?;
            expected_point.extend(p.first().copied());
            expected_point.extend(p.get(1..).unwrap_or(&[]).iter().rev().copied());
            if claims.key.point != expected_point
                || claims
                    .value
                    .as_ref()
                    .map(|c| c.point.as_slice())
                    .is_some_and(|point| point != expected_point)
                || claims.value.is_some() != instance.has_values
            {
                return Err(Error::Internal);
            }
            let (key0, value0) = row_zero(instance.table).ok_or(Error::Internal)?;
            if value0.is_some() != instance.has_values {
                return Err(Error::Internal);
            }
            let padding =
                Fp2::ONE - crate::selectors::lt(p, instance.entries).ok_or(Error::Internal)?;
            let mut key = padding * Fp2::from(key0);
            let mut value = padding * Fp2::from(value0.unwrap_or(Fp::ZERO));
            for block in &instance.blocks {
                let k = usize::try_from(block.variables).map_err(|_| Error::Internal)?;
                let weight = p
                    .get(k..)
                    .ok_or(Error::Internal)?
                    .iter()
                    .enumerate()
                    .try_fold(Fp2::ONE, |weight, (bit, &r)| {
                        let shift = block
                            .variables
                            .checked_add(u32::try_from(bit).map_err(|_| Error::Internal)?)
                            .ok_or(Error::Internal)?;
                        let high = block.offset.checked_shr(shift).ok_or(Error::Internal)? & 1;
                        Ok::<_, Error>(weight * if high == 0 { Fp2::ONE - r } else { r })
                    })?;
                key += weight * evaluate(&block.key, block.layer)?;
                if let Some(form) = &block.value {
                    value += weight * evaluate(form, block.layer)?;
                }
            }
            if key != claims.key.value
                || claims
                    .value
                    .as_ref()
                    .is_some_and(|claim| claim.value != value)
            {
                return Err(Error::Mismatch);
            }
        }
        let mut columns = vector(self.values)?;
        for input in &self.inputs {
            for (offset, read) in input.reads.iter().enumerate() {
                let mut point = vector(read.coordinates.len())?;
                for &coordinate in &read.coordinates {
                    point.push(
                        *leaf
                            .point
                            .get(usize::try_from(coordinate).map_err(|_| Error::Internal)?)
                            .ok_or(Error::Internal)?,
                    );
                }
                columns.push(ColumnClaim {
                    node: input.node,
                    column: read.column,
                    point,
                    value: *splits
                        .get(input.start.checked_add(offset).ok_or(Error::Internal)?)
                        .ok_or(Error::Internal)?,
                });
            }
        }
        let mut multiplicities = vector(self.instances.len())?;
        for (instance, claim) in self.instances.iter().zip(leaf.lookups.instances) {
            multiplicities.push((instance.table, claim.multiplicity));
        }
        Ok(Bound {
            columns,
            multiplicities,
            forests: leaf.forests,
            atoms: copied(splits.get(self.values..).ok_or(Error::Internal)?).map_err(own)?,
            point: LeafPoint::Shared(leaf.point),
        })
    }
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    #[test]
    fn original_first_read_identity_and_coordinates_are_required() {
        let layout = Layout {
            inputs: vec![Inputs {
                node: NodeId(7),
                reads: vec![Read {
                    column: Column::Input(0),
                    coordinates: vec![1],
                }],
                tokens: vec![None],
                start: 0,
            }],
            instances: Vec::new(),
            values: 1,
        };
        let bound = Bound {
            columns: vec![ColumnClaim {
                node: NodeId(7),
                column: Column::Input(0),
                point: vec![Fp2::ONE],
                value: Fp2::ZERO,
            }],
            multiplicities: Vec::new(),
            forests: Vec::new(),
            atoms: Vec::new(),
            point: LeafPoint::Shared(vec![Fp2::ZERO, Fp2::ONE]),
        };
        assert_eq!(layout.check_bound(&bound), Ok(()));
        for fault in 0..6 {
            let mut bad = bound.clone();
            match fault {
                0 => bad.columns.push(bad.columns[0].clone()),
                1 => bad.columns[0].node = NodeId(8),
                2 => bad.columns[0].column = Column::Output(0),
                3 => bad.columns[0].point.clear(),
                4 => bad.columns[0].point[0] = Fp2::ZERO,
                5 => bad.point = LeafPoint::Shared(vec![Fp2::ZERO]),
                _ => unreachable!(),
            }
            assert_eq!(layout.check_bound(&bad), Err(Error::Internal));
        }
    }

    fn f(v: u64) -> Fp2 {
        Fp2::from(Fp::reduce(v))
    }

    /// RFC-0045 'The leaf point': a tensor of `n` variables lies at `P[0..n]`, whatever its
    /// token axis, and a read at its coordinates of `P`.
    #[test]
    fn an_ungrouped_stage_leaves_its_claims_on_prefixes_of_p() {
        let p = LeafPoint::Shared(vec![f(1), f(2), f(3), f(4)]);
        let axis = Some(TokenAxis { low: 1, bits: 2 });
        assert_eq!(p.tensor(3, axis), Some(vec![f(1), f(2), f(3)]));
        assert_eq!(p.tensor(0, None), Some(vec![]));
        assert_eq!(p.tensor(5, None), None);
        assert_eq!(p.read(&[2, 0], axis), Some(vec![f(3), f(1)]));
        assert_eq!(p.read(&[4], None), None);
    }

    /// RFC-0054 'The merged point': `r(n, S)` is the merge's last `n` challenges, the variables
    /// outside `S` first, lowest first, then those in `S`, lowest first.
    #[test]
    fn a_grouped_stage_leaves_its_claims_at_the_merged_points() {
        // Challenges r_0 … r_5; g_T = 2.
        let r: Vec<Fp2> = (10..16).map(f).collect();
        let merged = LeafPoint::Merged {
            challenges: r.clone(),
            token_bits: 2,
        };
        // A tensor [n, d] of 5 variables, its token axis on variables 2..5: S = {2, 3}. The last
        // five challenges r_1 … r_5: variables 0, 1, 4 take r_1, r_2, r_3; S takes r_4, r_5.
        let axis = Some(TokenAxis { low: 2, bits: 3 });
        assert_eq!(
            merged.tensor(5, axis),
            Some(vec![r[1], r[2], r[4], r[5], r[3]])
        );
        // The same tensor read at its own coordinates of a block with that token axis.
        assert_eq!(merged.read(&[0, 1, 2, 3, 4], axis), merged.tensor(5, axis));
        // A tensor without a token axis, such as a channel's class: the last two, in order.
        assert_eq!(merged.tensor(2, None), Some(vec![r[4], r[5]]));
        assert_eq!(merged.read(&[0, 1], axis), Some(vec![r[4], r[5]]));
        // A row tensor [n] of 3 variables, its token axis lowest: S = {0, 1}.
        let rows = Some(TokenAxis { low: 0, bits: 3 });
        assert_eq!(merged.tensor(3, rows), Some(vec![r[4], r[5], r[3]]));
        // Without token bits, a layer grouping alone: the last n challenges, in order.
        let layers = LeafPoint::Merged {
            challenges: r.clone(),
            token_bits: 0,
        };
        assert_eq!(layers.tensor(3, rows), Some(vec![r[3], r[4], r[5]]));
        // No point beyond the merge's rounds, nor for a token axis outside the tensor or
        // shorter than g_T.
        assert_eq!(merged.tensor(7, None), None);
        assert_eq!(merged.tensor(4, axis), None);
        assert_eq!(merged.tensor(3, Some(TokenAxis { low: 0, bits: 1 })), None);
    }
}
