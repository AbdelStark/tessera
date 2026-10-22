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

/// The debts after checking the lookup split; not an acceptance verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "check all source/forest reductions and multiplicity openings, and the graph atom splits"]
pub struct Bound {
    /// Every distinct source, in site/layer/first-read order.
    pub columns: Vec<ColumnClaim>,
    /// Table multiplicities retain their original table-tree points and table identities.
    pub multiplicities: Vec<(TableId, Claim)>,
    /// All original product-forest leaf claims, at their own points.
    pub forests: Vec<ForestClaim>,
    /// Graph atom split values from the tail of the same absorbed vector.
    pub atoms: Vec<Fp2>,
    /// The shared lookup point from the tree transcript.
    pub point: Vec<Fp2>,
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
                                        reads.push(Read {
                                            column: read.column,
                                            coordinates: copied(&read.coordinates).map_err(own)?,
                                        });
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
            for (i, read) in input.reads.iter().enumerate() {
                let at = input.start.checked_add(i).ok_or(Error::Internal)?;
                let column = bound.columns.get(at).ok_or(Error::Internal)?;
                if column.node != input.node
                    || column.column != read.column
                    || column.point.len() != read.coordinates.len()
                {
                    return Err(Error::Internal);
                }
                for (&coordinate, value) in read.coordinates.iter().zip(&column.point) {
                    let coordinate = usize::try_from(coordinate).map_err(|_| Error::Internal)?;
                    if bound.point.get(coordinate) != Some(value) {
                        return Err(Error::Internal);
                    }
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
            point: leaf.point,
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
            point: vec![Fp2::ZERO, Fp2::ONE],
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
                5 => {
                    bad.point.pop();
                }
                _ => unreachable!(),
            }
            assert_eq!(layout.check_bound(&bad), Err(Error::Internal));
        }
    }
}
