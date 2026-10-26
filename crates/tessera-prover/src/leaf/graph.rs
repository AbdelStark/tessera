// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The leaf sites of a graph without a lowering, such as the anchor's AES-128-CTR circuit
//! (RFC-0072 'Gadgets and leaf sites' 'AES's lookups'): each `TableMap` and `Bitwise` node is a
//! lookup gadget, one site in node order with RFC-0008's key (RFC-0045 'Sites'), and
//! [`GraphLeafWitness`] lends the tensors its lookups read over their padded shapes.
//!
//! - **Keys.** A `Bitwise` into `Bitwise8` looks up `a + 2^8·b` with its output; a `TableMap`
//!   its input, or `pack(c, z) = (z + 2^{b−1}) + 2^b·c` for a `Requant` or `Silu` table, its
//!   classes read over the channel axis (RFC-0008 'Table kinds').
//! - **Public inputs.** A lookup that reads a public tensor, such as AES's counter blocks or its
//!   round constants, sends no split value for it ([`Read::public`](super::Read::public),
//!   RFC-0045 'The split'): the verifier evaluates it at the block's point.
//! - **Padded positions.** A block runs over its domain's padded shape, each position holding the
//!   lookup of what its node computes there ([`PaddedWitness`](crate::PaddedWitness)), such as
//!   `AesSbox(0) = 0x63` in a padded block; the multiplicities count them as any lookup.
//! - **No token groups.** A graph without a lowering has no token axis (RFC-0054), so its leaf
//!   stage runs ungrouped ([`LeafLayout::ungroupable`]).

use core::fmt;

use tessera_codec::SecretBuf;
use tessera_compiler::schedule::{LayerMap, NodeRef, has_leaf_site, public_tensors};
use tessera_compiler::{Gadget, Selection};
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_gadgets::{Affine, Lookup};
use tessera_ir::{Graph, Node, NodeId, Op, TensorId};
use tessera_tables::TableKind;

use super::{
    LeafError, LeafLayout, ModelColumn, Site, SiteGadget, SiteLookup, SiteTensors, internal,
};
use crate::stages::WitnessSource;

/// Why a graph's leaf stage runs ungrouped ([`LeafLayout::ungroupable`]).
const NO_TOKENS: &str = "a graph without a lowering has no token axis (RFC-0054 'The token \
                         axis'): its leaf stage runs ungrouped (RFC-0072 'Non-Goals')";

impl LeafLayout<ModelColumn> {
    /// The leaf layout of `graph`, a graph without a lowering, its layers `layers` and its gadgets
    /// `selection` (RFC-0045, RFC-0072 'Gadgets and leaf sites'). Each `TableMap` and `Bitwise`
    /// node is a site in node order, a template position once with one gadget per real layer,
    /// with one lookup into the table gadget selection gives it (`Gadget::Lookup`):
    ///
    /// - **`Bitwise`** into `Bitwise8`: the key `a + 2^8·b`, `Input(0)` and `Input(1)`, or
    ///   `Input(0)` twice when both operands are one tensor, which the split then sends once; the
    ///   value `Output(0)`;
    /// - **`TableMap`** into `Requant` or `Silu`: the key `(z + 2^{b−1}) + 2^b·c`, its classes
    ///   `Input(1)` read over the channel axis when the table has more than one; into a table of
    ///   any other kind with values: its input `Input(0)`; the value `Output(0)`.
    ///
    /// Each lookup's domain is its input's shape. A read of a public tensor ([`public_tensors`])
    /// is public: the split sends no value for it ([`Read::public`](super::Read::public)). The
    /// layout has no token axis, so it runs ungrouped ([`LeafLayout::ungroupable`]).
    ///
    /// # Errors
    ///
    /// [`LeafError::Internal`] (`ERR-INTERNAL`) for the first site that breaks a rule, in node
    /// order: a gadget other than a `TableMap` or a `Bitwise` with a leaf site, which a model's
    /// layout lays out ([`LeafLayout::model`]); a node without its `Gadget::Lookup` or table; a
    /// `Bitwise` into another kind than `Bitwise8`; a `TableMap` into a `Range`, `Bitwise8` or
    /// `Rope` table, or with other inputs than its table's classes need. Then the errors of
    /// [`LeafLayout::new`].
    pub fn graph(
        graph: &Graph,
        layers: &LayerMap,
        selection: &Selection,
    ) -> Result<Self, LeafError> {
        let mut sites = Vec::new();
        for node in &graph.nodes {
            if !has_leaf_site(&node.op) {
                continue;
            }
            let (site, nodes): (NodeRef, Vec<NodeId>) = match layers.position(node.id) {
                Some((0, p)) => (
                    NodeRef::Position(p),
                    (0..layers.count).map(|l| layers.node(p, l)).collect(),
                ),
                Some(_) => continue,
                None => (NodeRef::Node(node.id), vec![node.id]),
            };
            let gadgets = nodes
                .iter()
                .map(|id| lookup_site(graph, selection, &graph.nodes[id.0 as usize]))
                .collect::<Result<_, LeafError>>()?;
            sites.push(Site {
                node: site,
                layers: gadgets,
            });
        }
        let mut layout = Self::ungrouped(layers, sites, NO_TOKENS)?;
        let public = public_tensors(graph);
        let owners: Vec<Vec<NodeId>> = layout
            .sites()
            .iter()
            .map(|site| {
                (0..site.layers.len())
                    .map(|l| owner(layers, site.node, l))
                    .collect()
            })
            .collect();
        layout.mark_public(|s, l, read| {
            let node = &graph.nodes[owners[s][l].0 as usize];
            tensor(read.term, node)
                .is_some_and(|t| public.get(t.0 as usize).copied().unwrap_or(false))
        });
        Ok(layout)
    }
}

/// The site's gadget at `node`: its one lookup, with RFC-0008's key and its output as the value.
fn lookup_site(
    graph: &Graph,
    selection: &Selection,
    node: &Node,
) -> Result<SiteGadget<ModelColumn>, LeafError> {
    let table = match node.op {
        Op::TableMap { table } | Op::Bitwise { table } => table,
        _ => {
            return internal(
                "a graph's leaf sites are TableMap and Bitwise lookups; a model's layout lays out \
                 the other gadgets",
            );
        }
    };
    let selected = selection.of(node.id).map(|(_, gadget)| gadget);
    if selected != Some(Gadget::Lookup { table })
        || selection.reads(node.id) != Some(core::slice::from_ref(&table))
    {
        return internal("gadget selection gives a lookup node its one table");
    }
    let spec = selection
        .tables
        .iter()
        .find(|spec| spec.id == table)
        .ok_or(LeafError::Internal {
            rule: "a lookup's table is the selection's",
        })?;
    let domain = node
        .inputs
        .first()
        .and_then(|t| graph.tensors.get(t.0 as usize))
        .map(|t| t.shape.clone())
        .ok_or(LeafError::Internal {
            rule: "a lookup reads its input",
        })?;
    let one = Goldilocks::ONE;
    let mut axes = Vec::new();
    let key = match (&node.op, &spec.kind) {
        (Op::Bitwise { .. }, TableKind::Bitwise8 { .. }) => {
            // Both operands one tensor: one read, sent once (RFC-0045 'What is sent').
            let second = match node.inputs.get(1) {
                Some(b) if Some(b) == node.inputs.first() => ModelColumn::Input(0),
                Some(_) => ModelColumn::Input(1),
                None => return internal("a Bitwise reads two operands"),
            };
            Affine {
                constant: Goldilocks::ZERO,
                terms: vec![
                    (one, ModelColumn::Input(0)),
                    (Goldilocks::from_u64_reduced(1 << 8), second),
                ],
            }
        }
        (Op::Bitwise { .. }, _) => return internal("a Bitwise looks up a Bitwise8 table"),
        (
            Op::TableMap { .. },
            TableKind::Requant { b, r_by_class, .. } | TableKind::Silu { b, r_by_class, .. },
        ) => {
            let classed = r_by_class.len() > 1;
            if node.inputs.len() != 1 + usize::from(classed) {
                return internal("a TableMap reads its classes exactly when its table has several");
            }
            let offset = u32::from(*b)
                .checked_sub(1)
                .and_then(|e| 1u64.checked_shl(e));
            let scale = 1u64.checked_shl(u32::from(*b));
            let (Some(offset), Some(scale)) = (offset, scale) else {
                return internal("a packed key's offset and class scale fit 64 bits");
            };
            let mut terms = vec![(one, ModelColumn::Input(0))];
            if classed {
                terms.push((Goldilocks::from_u64_reduced(scale), ModelColumn::Input(1)));
                axes.push((ModelColumn::Input(1), vec![domain.len().saturating_sub(1)]));
            }
            Affine {
                constant: Goldilocks::from_u64_reduced(offset),
                terms,
            }
        }
        (
            Op::TableMap { .. },
            TableKind::Range { .. } | TableKind::Bitwise8 { .. } | TableKind::Rope { .. },
        ) => {
            return internal("a TableMap looks up a table of one operand with values");
        }
        (Op::TableMap { .. }, _) => {
            if node.inputs.len() != 1 {
                return internal("a TableMap reads its classes exactly when its table has several");
            }
            Affine {
                constant: Goldilocks::ZERO,
                terms: vec![(one, ModelColumn::Input(0))],
            }
        }
        _ => return internal("a graph's leaf sites are TableMap and Bitwise lookups"),
    };
    Ok(SiteGadget {
        lookups: vec![SiteLookup {
            lookup: Lookup {
                table,
                key,
                value: Some(Affine {
                    constant: Goldilocks::ZERO,
                    terms: vec![(one, ModelColumn::Output(0))],
                }),
            },
            domain,
            axes,
            token: None,
        }],
    })
}

/// The node of `site` in its layer `layer`.
fn owner(layers: &LayerMap, site: NodeRef, layer: usize) -> NodeId {
    match site {
        NodeRef::Node(id) => id,
        NodeRef::Position(p) => layers.node(p, u32::try_from(layer).unwrap_or(u32::MAX)),
    }
}

/// The tensor that `column` names at `node`: an input or an output.
fn tensor(column: ModelColumn, node: &Node) -> Option<TensorId> {
    match column {
        ModelColumn::Input(i) => node.inputs.get(usize::from(i)).copied(),
        ModelColumn::Output(i) => node.outputs.get(usize::from(i)).copied(),
        _ => None,
    }
}

/// The tensors a graph's leaf sites read ([`LeafLayout::graph`]), over their padded shapes: each
/// site's in each of its layers, read by read, a public one among them. A tensor that a lookup
/// non-zero at padding reaches holds what its operation computes at its padded positions, which
/// its source gives ([`WitnessSource::padded`], RFC-0072 'Padded positions'); any other is its
/// values padded with zeros.
///
/// The tables are secret-class (05 §3), in `SecretBuf`s that clear their memory, lent for the
/// leaf stage's proof ([`tensors`](Self::tensors)). The debug form shows the counts, never a
/// value.
pub struct GraphLeafWitness {
    tensors: Vec<Vec<Reads>>,
}

/// One site's tensors in one layer, read by read, each over its padded shape.
type Reads = Vec<(ModelColumn, SecretBuf<i64>)>;

/// Shows the sites, never a value (05 §3).
impl fmt::Debug for GraphLeafWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphLeafWitness")
            .field("sites", &self.tensors.len())
            .finish_non_exhaustive()
    }
}

impl GraphLeafWitness {
    /// The tensors `layout`'s sites read in `graph`, its layers `layers`, from `witness`.
    ///
    /// # Errors
    ///
    /// [`LeafError::Internal`] (`ERR-INTERNAL`) for a read that names no tensor of its node, a
    /// tensor `witness` lacks, or one whose padded table is not `2^k` entries for its read's `k`
    /// coordinates; [`LeafError::Memory`] when a table does not fit.
    pub fn new(
        layout: &LeafLayout<ModelColumn>,
        graph: &Graph,
        layers: &LayerMap,
        witness: &(impl WitnessSource + ?Sized),
    ) -> Result<Self, LeafError> {
        let mut tensors = Vec::with_capacity(layout.sites().len());
        for (s, site) in layout.sites().iter().enumerate() {
            let mut per_layer = Vec::with_capacity(site.layers.len());
            for l in 0..site.layers.len() {
                let node = graph
                    .nodes
                    .get(owner(layers, site.node, l).0 as usize)
                    .ok_or(LeafError::Internal {
                        rule: "a site's node is the graph's",
                    })?;
                let reads = layout.reads(s, l).ok_or(LeafError::Internal {
                    rule: "the layout reads every site's tensors",
                })?;
                let mut columns = Vec::with_capacity(reads.len());
                for read in reads {
                    let t = tensor(read.term, node).ok_or(LeafError::Internal {
                        rule: "a graph site reads its node's inputs and outputs",
                    })?;
                    let table = padded_table(graph, witness, t)?;
                    let entries = u32::try_from(read.coords.len())
                        .ok()
                        .and_then(|vars| 1usize.checked_shl(vars));
                    if Some(table.len()) != entries {
                        return internal("a read's tensor holds 2^k entries, padded per dimension");
                    }
                    columns.push((read.term, table));
                }
                per_layer.push(columns);
            }
            tensors.push(per_layer);
        }
        Ok(Self { tensors })
    }

    /// The tensors, lent for the leaf stage's proof, site by site and layer by layer
    /// ([`prepare_leaf`](super::prepare_leaf)).
    pub fn tensors(&self) -> Vec<Vec<SiteTensors<'_, ModelColumn>>> {
        self.tensors
            .iter()
            .map(|layers| {
                layers
                    .iter()
                    .map(|columns| {
                        SiteTensors::new(
                            columns
                                .iter()
                                .map(|(term, table)| (*term, &table[..]))
                                .collect(),
                        )
                    })
                    .collect()
            })
            .collect()
    }
}

/// `tensor` over its padded shape: the source's padded values, or its values padded with zeros.
fn padded_table(
    graph: &Graph,
    witness: &(impl WitnessSource + ?Sized),
    tensor: TensorId,
) -> Result<SecretBuf<i64>, LeafError> {
    let shape = &graph
        .tensors
        .get(tensor.0 as usize)
        .ok_or(LeafError::Internal {
            rule: "a site's tensor is the graph's",
        })?
        .shape;
    let vars: u32 = shape.iter().map(|&len| super::layout::axis_vars(len)).sum();
    let len = 1usize.checked_shl(vars).ok_or(LeafError::Memory)?;
    let mut table = SecretBuf::try_with_capacity(len).map_err(|_| LeafError::Memory)?;
    if let Some(values) = witness.padded(tensor) {
        if values.len() != len {
            return internal("a tensor's padded values fill its padded shape");
        }
        table.extend(values.iter().copied());
        return Ok(table);
    }
    let values = witness.values(tensor).ok_or(LeafError::Internal {
        rule: "the witness holds every tensor a site reads",
    })?;
    if values.len() != shape.iter().map(|&len| len as usize).product::<usize>() {
        return internal("a tensor holds its shape's number of values");
    }
    table.resize(len, 0);
    for (flat, &v) in values.iter().enumerate() {
        let (mut rest, mut index, mut shift) = (flat, 0usize, 0u32);
        for &d in shape.iter().rev() {
            index |= (rest % d as usize) << shift;
            rest /= d as usize;
            shift += super::layout::axis_vars(d);
        }
        table[index] = v;
    }
    Ok(table)
}
