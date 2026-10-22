// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A tile's values (RFC-0041 'Tiles'): the elements of its shares, its gadget's committed values,
//! and its lookups' keys and values (RFC-0009), assembled from its node's inputs and outputs.
//!
//! - **What a tile is.** [`Target`] checks a [`SequenceTile`] against the lowering from public
//!   data alone, as `ExecutionPlan::expand` derives it: its node, its KV head and query block, and
//!   the shares RFC-0041 gives it. Anything else is refused (`ERR-USAGE`).
//! - **Gadgets.** Requantization and root witnesses come from `tessera-gadgets`; root columns
//!   follow RFC-0050's grouped array order and public limb count. Softmax adds its maximum,
//!   slack limbs and every virtual-top lookup, with masked Exp key `K_0` (RFC-0030).
//! - **Multiplicities.** A table tile recounts every contributing node's lookups, including
//!   per-dimension padding and the combined column's row-0 tail (RFC-0008, RFC-0045). Public
//!   sizes choose dense counts or fixed-capacity weighted updates; implicit zero positions remain
//!   in the full canonical column. A bounded chunk reader never skips commitment positions.
//! - **Lookups.** A key is the row it names, key `j` at row `j` of every table (RFC-0008). A value
//!   is always one of the tile's committed tensors, `y`, `g` or `e`, which the lookup lends.
//!
//! The values are secret-class (05 §3): every one lives in a `SecretBuf`, the debug forms show
//! shapes and identifiers, and a tile lends its values to the code that holds it. Whether every key
//! names a row of its table, the one outcome that depends on the values, is folded over the
//! elements and reported after the last (INV-ARCH-2).

use core::fmt;
use core::ops::Range;

use tessera_codec::SecretBuf;
use tessera_compiler::{AuxColumn, Gadget, Selection, SequenceTile, Share, Tile};
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_gadgets::{Affine, Committed, Constraints, Requant, Term};
use tessera_ir::{CommitmentClass, Node, NodeKind, Op, OpClass, TableId, TensorId};
use tessera_tables::{Table, TableKind};

mod auxiliary;

use auxiliary::{Prepared, requant_role};

use crate::WitnessError;
use crate::engine::Operand;
use crate::multiplicity::{Multiplicities, MultiplicityColumn};
use crate::replay::Rows;
use crate::structure::{Core, Structure};

/// The refusals of a tile that is not one of its sequence's (RFC-0041 'Tiles').
const NOT_A_NODE: &str = "a tile names a node of the lowering's graph";
const NOT_ATTENTION: &str = "an attention tile names a KV head of a layer's SoftmaxRow";
const NOT_A_BLOCK: &str = "an attention tile's query block is one of its sequence's";
const NO_BLOCK: &str = "only an attention tile has a query block";
const NOT_READ: &str = "a table tile names a table the gadgets read";
const NOT_ITS_SHARES: &str = "a tile's shares are those RFC-0041 gives it";

/// Every lookup's key names a row of its table (RFC-0008, INV-IR-2).
const NAMES_A_ROW: &str = "every lookup's key names a row of its table (INV-IR-2)";

/// The refusal of a request that breaks `rule`.
fn usage<T>(rule: &'static str) -> Result<T, WitnessError> {
    Err(WitnessError::Usage { rule })
}

/// One tile's values (RFC-0041 'Tiles'): its shares' elements, its gadget's committed values beyond
/// its node's outputs, and its gadget's lookups at its elements.
///
/// Every value is secret-class (05 §3), in a `SecretBuf` that clears its memory (OBS-6), and the
/// tile lends them, as slices, to the code that holds it. Their holder puts no value, and nothing
/// derived from one, in a log, an error, a metric or a report; hands them only to code that
/// `secret-access.toml` lists (OBS-1); and copies none out, since the clearing cannot reach a copy.
/// The debug form shows the shapes, never a value.
pub struct TileWitness {
    shares: Vec<SecretBuf<i64>>,
    gadget: Vec<(AuxColumn, SecretBuf<i64>)>,
    lookups: Vec<Lookup>,
    multiplicities: Option<MultiplicityColumn>,
}

/// A lookup's columns: its keys, and the committed tensor of the tile that holds its values.
struct Lookup {
    table: TableId,
    keys: SecretBuf<u32>,
    values: Option<Column>,
}

/// A committed tensor of the tile: a share, or one of the gadget's values.
#[derive(Clone, Copy)]
enum Column {
    Share(usize),
    Gadget(usize),
}

impl TileWitness {
    /// A tile before its values are assembled.
    pub(crate) fn empty() -> Self {
        Self {
            shares: Vec::new(),
            gadget: Vec::new(),
            lookups: Vec::new(),
            multiplicities: None,
        }
    }

    /// The tokens' tile: the tokens alone.
    pub(crate) fn tokens(tokens: SecretBuf<i64>) -> Self {
        Self {
            shares: vec![tokens],
            gadget: Vec::new(),
            lookups: Vec::new(),
            multiplicities: None,
        }
    }

    /// The table tile's multiplicities, in table-row order, including zero-valued table-tree
    /// padding. Counts include per-dimension lookup padding and the lookup column's row-0 tail
    /// (RFC-0045); `None` for a tile other than a table. Secret-class, lent under this tile's rules.
    pub fn multiplicities(&self) -> Option<&MultiplicityColumn> {
        self.multiplicities.as_ref()
    }

    pub(crate) fn table(column: MultiplicityColumn) -> Self {
        Self {
            multiplicities: Some(column),
            ..Self::empty()
        }
    }

    /// The elements of each of the tile's shares, in the tile's order: for each share, the
    /// elements of its tensor in the share's ranges, row-major over them, outermost axis first
    /// (RFC-0041 'Tiles'). The values are lent under the rules of [`TileWitness`].
    pub fn shares(&self) -> &[SecretBuf<i64>] {
        &self.shares
    }

    /// The gadget's real auxiliary columns beyond graph outputs, named and ordered by RFC-0050's
    /// [`AuxColumn`]. Requantization/root columns use the node's element domain; Softmax maximum
    /// and slack limbs use its query-row domain. Virtual top limbs occur only as lookup keys.
    /// Values are lent under the rules of [`TileWitness`].
    pub fn gadget(&self) -> &[(AuxColumn, SecretBuf<i64>)] {
        &self.gadget
    }

    /// The gadget's lookups at the tile's elements, in RFC-0009's order: for each, element `x`'s
    /// `(key, value)` is a row of its table. The elements are those of the gadget's input, row-major,
    /// for a node's tile, and those of the attention tile's exponentials, in its share's order.
    pub fn lookups(&self) -> impl ExactSizeIterator<Item = TileLookup<'_>> + '_ {
        self.lookups.iter().map(|lookup| TileLookup {
            table: lookup.table,
            keys: &lookup.keys,
            values: lookup.values.map(|column| match column {
                Column::Share(k) => &self.shares[k][..],
                Column::Gadget(k) => &self.gadget[k].1[..],
            }),
        })
    }

    /// The bytes its buffers hold: the memory the tile takes beside the source that regenerated it.
    pub fn bytes(&self) -> usize {
        let wide = self.shares.iter().map(SecretBuf::capacity).sum::<usize>()
            + self.gadget.iter().map(|(_, g)| g.capacity()).sum::<usize>();
        let keys: usize = self.lookups.iter().map(|l| l.keys.capacity()).sum();
        wide * size_of::<i64>()
            + keys * size_of::<u32>()
            + self
                .multiplicities
                .as_ref()
                .map_or(0, MultiplicityColumn::bytes)
    }
}

/// Shows the shapes: each share's and gadget value's length, and each lookup's table and length.
impl fmt::Debug for TileWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shares: Vec<usize> = self.shares.iter().map(|s| s.len()).collect();
        let gadget: Vec<(AuxColumn, usize)> =
            self.gadget.iter().map(|(c, g)| (*c, g.len())).collect();
        f.debug_struct("TileWitness")
            .field("shares", &shares)
            .field("gadget", &gadget)
            .field(
                "multiplicities",
                &self.multiplicities.as_ref().map(MultiplicityColumn::rows),
            )
            .field("lookups", &self.lookups().collect::<Vec<_>>())
            .finish()
    }
}

/// One lookup of a tile's gadget at each of the tile's elements (RFC-0009): element `x`'s key
/// `keys()[x]`, the row it names, and its value `values()[x]` form a row of `table()`.
///
/// The keys and values are secret-class (05 §3), lent from the tile under its rules. The debug form
/// shows the table and the lengths, never a key or a value.
#[derive(Clone, Copy)]
pub struct TileLookup<'a> {
    table: TableId,
    keys: &'a [u32],
    values: Option<&'a [i64]>,
}

impl<'a> TileLookup<'a> {
    /// The table.
    pub fn table(&self) -> TableId {
        self.table
    }

    /// Each element's key: the row of the table it names, key `j` at row `j` (RFC-0008).
    pub fn keys(&self) -> &'a [u32] {
        self.keys
    }

    /// Each element's value, `None` for a table without values (`Range`): one of the tile's
    /// committed tensors, `y`, `g` or `e`.
    pub fn values(&self) -> Option<&'a [i64]> {
        self.values
    }
}

/// Shows the table and the lengths (05 §3).
impl fmt::Debug for TileLookup<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TileLookup")
            .field("table", &self.table)
            .field("keys", &self.keys.len())
            .field("values", &self.values.map(<[i64]>::len))
            .finish()
    }
}

/// A tile, checked against its sequence from public data alone (RFC-0041 'Tiles').
pub(crate) enum Target {
    /// The tokens.
    Tokens,
    /// A table's multiplicities: no graph tensor.
    Table(TableId),
    /// The node of this index.
    Node(u32),
    /// Rows of an attention core: its query heads `heads` at the queries `queries` the sequence
    /// holds, every key with them.
    Attention {
        core: usize,
        heads: Range<usize>,
        queries: Range<usize>,
    },
}

/// Whether a tensor of `kind` is committed in the advice: advice or a lookup's output (RFC-0041
/// 'What is committed').
fn committed(kind: NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::Committed {
            commitment: CommitmentClass::Advice
        } | NodeKind::LookupOutput { .. }
    )
}

impl Structure<'_> {
    /// All of `tensor`, when it is committed in the advice.
    pub(crate) fn whole(&self, tensor: TensorId) -> Option<Share> {
        let spec = self.graph().tensors.get(tensor.0 as usize)?;
        committed(spec.kind).then(|| Share {
            tensor,
            ranges: spec.shape.iter().map(|&len| 0..len).collect(),
        })
    }

    /// What `tile` is, after checking that it is one of the sequence's tiles, as
    /// `ExecutionPlan::expand` gives them, with the shares RFC-0041 gives it.
    ///
    /// # Errors
    ///
    /// [`WitnessError::Usage`] for the first rule broken: a node past the graph; an attention tile
    /// on a node that is no layer's `SoftmaxRow`, or on a KV head past `n_kv_heads`, or without
    /// one of the sequence's query blocks; a query block on another tile; a table tile on a table
    /// no gadget reads; then shares other than RFC-0041's.
    pub(crate) fn target(
        &self,
        tile: &SequenceTile,
        read: &[TableId],
    ) -> Result<Target, WitnessError> {
        let graph = self.graph();
        let node_of = |node: u32| {
            graph
                .nodes
                .get(node as usize)
                .ok_or(WitnessError::Usage { rule: NOT_A_NODE })
        };
        let no_block = || match tile.block {
            Some(_) => usage(NO_BLOCK),
            None => Ok(()),
        };
        let (target, shares): (Target, Vec<Share>) = match tile.tile {
            Tile::Tokens => {
                no_block()?;
                (
                    Target::Tokens,
                    self.whole(self.lowering.tokens).into_iter().collect(),
                )
            }
            Tile::Node { node } => {
                let node = node_of(node)?;
                no_block()?;
                if self.core_of(node.id.0).is_some()
                    || (node.op.class() != OpClass::Gadget
                        && !node.outputs.iter().any(|&tensor| {
                            graph
                                .tensors
                                .get(tensor.0 as usize)
                                .is_some_and(|t| committed(t.kind))
                        }))
                {
                    return usage("a node tile commits a value and is not a layer's SoftmaxRow");
                }
                let shares = node.outputs.iter().filter_map(|&t| self.whole(t)).collect();
                (Target::Node(node.id.0), shares)
            }
            Tile::Attention { node, kv_head } => {
                let node = node_of(node)?;
                let (Some(core), Some(heads)) =
                    (self.core_of(node.id.0), self.query_heads(kv_head))
                else {
                    return usage(NOT_ATTENTION);
                };
                // Its scores are `[n_heads, n, n]`, as `ExecutionPlan::expand` requires.
                if self.cores[core].heads != self.heads() {
                    return usage(NOT_ATTENTION);
                }
                let n = self.positions();
                let padded = n.next_power_of_two();
                let Some(block) = tile.block.as_ref() else {
                    return usage(NOT_A_BLOCK);
                };
                let (start, end) = (block.queries.start as usize, block.queries.end as usize);
                let width = end.saturating_sub(start);
                let aligned = width.is_power_of_two()
                    && width <= padded
                    && start == block.index as usize * width
                    && end <= padded;
                if !aligned {
                    return usage(NOT_A_BLOCK);
                }
                let queries = start.min(n)..end.min(n);
                let as_u32 = |r: &Range<usize>| r.start as u32..r.end as u32;
                let shares = node
                    .outputs
                    .iter()
                    .filter_map(|&t| {
                        let mut share = self.whole(t)?;
                        if let [h, q, ..] = share.ranges.as_mut_slice() {
                            *h = as_u32(&heads);
                            *q = as_u32(&queries);
                        }
                        Some(share)
                    })
                    .collect();
                (
                    Target::Attention {
                        core,
                        heads,
                        queries,
                    },
                    shares,
                )
            }
            Tile::GreedyMaximum { .. } => {
                return usage(
                    "greedy certificates require the runtime witness source (draft RFC-0053)",
                );
            }
            Tile::Table { table } => {
                no_block()?;
                if !read.contains(&TableId(table)) {
                    return usage(NOT_READ);
                }
                (Target::Table(TableId(table)), Vec::new())
            }
        };
        if tile.shares != shares {
            return usage(NOT_ITS_SHARES);
        }
        Ok(target)
    }
}

/// A requantization's gadget, with its constraints.
struct RequantGadget {
    requant: Requant,
    constraints: Constraints<Goldilocks>,
    /// Each table its lookups read, with its rows.
    rows: Vec<(TableId, u64)>,
}

/// The gadgets a tile's values come from (RFC-0009): each requantization's, built from the
/// selection and the tables it names, and the tables some gadget reads.
pub(crate) struct Gadgets {
    requant: Vec<Option<RequantGadget>>,
    requests: Vec<Vec<TableId>>,
    auxiliary: Vec<Option<Prepared>>,
    pub(crate) read: Vec<TableId>,
    incomplete: Vec<TableId>,
}

impl Gadgets {
    /// The gadgets `selection` gives the nodes of `structure`'s graph.
    ///
    /// # Errors
    ///
    /// [`WitnessError::Internal`] for a requantization without its gadget;
    /// [`WitnessError::Gadget`] for a gadget that refuses its tables.
    pub(crate) fn new(
        structure: &Structure<'_>,
        selection: &Selection,
        max_context: u32,
    ) -> Result<Self, WitnessError> {
        let graph = structure.graph();
        let mut requant = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            let gadget = match node.op {
                Op::Requant { .. } => {
                    let (_, gadget) = selection.of(node.id).ok_or(WitnessError::Internal {
                        rule: "the selection gives every requantization its gadget",
                    })?;
                    let built = Requant::new(&gadget, &selection.tables)?;
                    let rows = built
                        .tables()
                        .into_iter()
                        .map(|t| (t.spec().id, t.row_count()))
                        .collect();
                    Some(RequantGadget {
                        constraints: built.constraints::<Goldilocks>(),
                        requant: built,
                        rows,
                    })
                }
                _ => None,
            };
            requant.push(gadget);
        }
        let mut read = Vec::new();
        for &(node, _) in &selection.gadgets {
            read.extend_from_slice(selection.reads(node).ok_or(WitnessError::Internal {
                rule: "every selected gadget has its ordered lookup requests",
            })?);
        }
        read.sort_unstable();
        read.dedup();
        let mut auxiliary = Vec::with_capacity(graph.nodes.len());
        let mut incomplete = Vec::new();
        for node in &graph.nodes {
            auxiliary.push(Prepared::new(node, selection, max_context)?);
            if matches!(
                selection.of(node.id),
                Some((_, Gadget::Argmax { .. } | Gadget::InverseCdf))
            ) {
                incomplete.extend_from_slice(selection.reads(node.id).unwrap_or(&[]));
            }
        }
        Ok(Self {
            requant,
            auxiliary,
            requests: graph
                .nodes
                .iter()
                .map(|node| selection.reads(node.id).unwrap_or(&[]).to_vec())
                .collect(),
            read,
            incomplete,
        })
    }

    /// A node's tile from the node's inputs `inputs` and its outputs `outputs`: its shares, the
    /// committed outputs, then its gadget's values and lookups.
    pub(crate) fn node_tile(
        &self,
        structure: &Structure<'_>,
        node: &Node,
        inputs: &[Operand<'_>],
        outputs: Vec<SecretBuf<i64>>,
        shares: &[Share],
    ) -> Result<TileWitness, WitnessError> {
        let mut outputs: Vec<Option<SecretBuf<i64>>> = outputs.into_iter().map(Some).collect();
        let mut tile = TileWitness::empty();
        // The share of each output, by output.
        let mut share_of = vec![None; node.outputs.len()];
        for share in shares {
            let j = node.outputs.iter().position(|&t| t == share.tensor).ok_or(
                WitnessError::Usage {
                    rule: NOT_ITS_SHARES,
                },
            )?;
            let values = outputs[j].take().ok_or(WitnessError::Usage {
                rule: NOT_ITS_SHARES,
            })?;
            share_of[j] = Some(tile.shares.len());
            tile.shares.push(values);
        }
        let input = |k: usize| -> Result<&[i64], WitnessError> {
            match inputs.get(k) {
                Some(Operand::Values(values)) => Ok(*values),
                _ => Err(WitnessError::Internal {
                    rule: "a gadget reads its node's inputs as values",
                }),
            }
        };
        let output = |j: usize| {
            share_of
                .get(j)
                .copied()
                .flatten()
                .ok_or(WitnessError::Internal {
                    rule: "a gadget's output is committed in its tile",
                })
        };
        let graph = structure.graph();
        let channels = |k: usize| {
            graph.tensors[node.inputs[k].0 as usize]
                .shape
                .last()
                .map_or(1, |&c| c as usize)
        };
        match &node.op {
            Op::Requant { .. } => {
                let gadget = self
                    .requant
                    .get(node.id.0 as usize)
                    .and_then(Option::as_ref)
                    .ok_or(WitnessError::Internal {
                        rule: "the selection gives every requantization its gadget",
                    })?;
                let z = input(0)?;
                let classes = match node.inputs.len() > 1 {
                    true => Some(per_element(
                        node.inputs[0],
                        input(1)?,
                        channels(0),
                        z.len(),
                    )?),
                    false => None,
                };
                self.requant_tile(
                    node.outputs[0],
                    gadget,
                    z,
                    classes.as_deref(),
                    output(0)?,
                    &mut tile,
                )?;
            }
            Op::TableMap { table } => {
                let t = structure.engine.lookup(*table)?;
                let x = input(0)?;
                let classes = match node.inputs.len() > 1 {
                    true => Some(per_element(
                        node.inputs[0],
                        input(1)?,
                        channels(0),
                        x.len(),
                    )?),
                    false => None,
                };
                let keys = keys(node.outputs[0], x.len(), t.row_count(), |i| {
                    t.index(x[i], classes.as_ref().map_or(0, |c| c[i]))
                })?;
                tile.lookups.push(Lookup {
                    table: *table,
                    keys,
                    values: Some(Column::Share(output(0)?)),
                });
            }
            Op::Bitwise { table } => {
                let t = structure.engine.lookup(*table)?;
                let (a, b) = (input(0)?, input(1)?);
                // RFC-0008: the key of a Bitwise8 lookup is a + 2^8 · b.
                let keys = keys(node.outputs[0], a.len(), t.row_count(), |i| {
                    t.index(a[i] + (b[i] << 8), 0)
                })?;
                tile.lookups.push(Lookup {
                    table: *table,
                    keys,
                    values: Some(Column::Share(output(0)?)),
                });
            }
            Op::RsqrtRow { .. } => {
                self.prepared(node.id.0)?.root_tile(
                    node.outputs[0],
                    input(0)?,
                    output(0)?,
                    &mut tile,
                )?;
            }
            Op::SoftmaxRow { .. } => {
                self.prepared(node.id.0)?.softmax_tile(
                    node.outputs[0],
                    input(0)?,
                    channels(0),
                    |row| row % channels(0) + 1,
                    output(0)?,
                    output(1)?,
                    &mut tile,
                )?;
            }
            _ => {}
        }
        Ok(tile)
    }

    /// A requantization's committed values beyond `y` and its lookups, into `tile`, whose share
    /// `y` holds the node's output `tensor`.
    fn requant_tile(
        &self,
        tensor: TensorId,
        gadget: &RequantGadget,
        z: &[i64],
        classes: Option<&[u8]>,
        y: usize,
        tile: &mut TileWitness,
    ) -> Result<(), WitnessError> {
        let witness = gadget.requant.witness(z, classes)?;
        let mut same = true;
        for (held, values) in witness.into_tensors() {
            if held == Committed::Y {
                // The gadget's output is D's: the engine's, which the share holds. Folded over
                // every element, so that no comparison stops early (INV-ARCH-2).
                same &= values.len() == tile.shares[y].len();
                same &= values
                    .iter()
                    .zip(tile.shares[y].iter())
                    .fold(true, |all, (a, b)| all & (a == b));
            } else {
                tile.gadget.push((requant_role(held)?, values));
            }
        }
        if !same {
            return Err(WitnessError::Internal {
                rule: "a requantization gadget's output is the engine's",
            });
        }
        for lookup in &gadget.constraints.lookups {
            let rows = gadget
                .rows
                .iter()
                .find(|(t, _)| *t == lookup.table)
                .map(|&(_, rows)| rows)
                .ok_or(WitnessError::Internal {
                    rule: "a gadget's lookups read its tables",
                })?;
            // Each term's column, once for the whole lookup.
            let column = |term: Term| -> Result<Option<&[i64]>, WitnessError> {
                Ok(match term {
                    Term::Input => Some(z),
                    Term::Class => None,
                    Term::Committed(Committed::Y) => Some(&tile.shares[y][..]),
                    Term::Committed(c) => Some(
                        tile.gadget
                            .iter()
                            .find(|(g, _)| requant_role(c).is_ok_and(|role| *g == role))
                            .map(|(_, values)| &values[..])
                            .ok_or(WitnessError::Internal {
                                rule: "a lookup reads its gadget's committed values",
                            })?,
                    ),
                })
            };
            let terms = lookup
                .key
                .terms
                .iter()
                .map(|&(_, term)| column(term).map(|values| (term, values)))
                .collect::<Result<Vec<_>, _>>()?;
            let at = |term: Term, i: usize| -> Goldilocks {
                match terms.iter().find(|(t, _)| *t == term) {
                    Some((Term::Class, _)) => {
                        Goldilocks::from_u64_reduced(u64::from(classes.map_or(0, |c| c[i])))
                    }
                    Some((_, Some(values))) => Goldilocks::from_i64(values[i]),
                    _ => Goldilocks::ZERO,
                }
            };
            let keys = keys_of(tensor, &lookup.key, z.len(), rows, &at)?;
            let values = match &lookup.value {
                Some(form) => Some(value_column(form, y, &tile.gadget)?),
                None => None,
            };
            tile.lookups.push(Lookup {
                table: lookup.table,
                keys,
                values,
            });
        }
        Ok(())
    }

    /// An attention tile from its rows: the shares of its exponentials and reciprocals, and its
    /// lookups into the `Exp` table at every position of its rows, the queries `queries` of each
    /// query head after another.
    pub(crate) fn attention_tile(
        &self,
        _structure: &Structure<'_>,
        core: &Core,
        rows: Rows,
        queries: Range<usize>,
    ) -> Result<TileWitness, WitnessError> {
        let Rows { scores, e, r } = rows;
        let mut tile = TileWitness {
            shares: vec![e, r],
            gadget: Vec::new(),
            lookups: Vec::new(),
            multiplicities: None,
        };
        let held = queries.len().max(1);
        self.prepared(core.softmax)?.softmax_tile(
            core.e,
            &scores,
            core.n,
            |row| queries.start + row % held + 1,
            0,
            1,
            &mut tile,
        )?;
        Ok(tile)
    }
}

/// Each element's class, its output channel's: `classes[i mod channels]`, as the engine reads it.
fn per_element(
    tensor: TensorId,
    classes: &[i64],
    channels: usize,
    len: usize,
) -> Result<SecretBuf<u8>, WitnessError> {
    let mut column =
        SecretBuf::try_with_capacity(len).map_err(|_| WitnessError::Memory { tensor })?;
    let mut fits = classes.len() == channels;
    for i in 0..len {
        let class = classes.get(i % channels.max(1)).copied().unwrap_or(0);
        let byte = u8::try_from(class);
        fits &= byte.is_ok();
        column.push(byte.unwrap_or(0));
    }
    match fits {
        true => Ok(column),
        false => Err(WitnessError::Internal {
            rule: "a channel's class indexes its site's shift classes",
        }),
    }
}

/// `len` keys, the row each element's lookup names, below `rows`; whether each names one is folded
/// and reported after the last.
fn keys(
    tensor: TensorId,
    len: usize,
    rows: u64,
    row: impl Fn(usize) -> Option<u64>,
) -> Result<SecretBuf<u32>, WitnessError> {
    let mut keys =
        SecretBuf::try_with_capacity(len).map_err(|_| WitnessError::Memory { tensor })?;
    let mut fits = true;
    for i in 0..len {
        let named = row(i)
            .filter(|&r| r < rows)
            .and_then(|r| u32::try_from(r).ok());
        fits &= named.is_some();
        keys.push(named.unwrap_or(0));
    }
    match fits {
        true => Ok(keys),
        false => Err(WitnessError::Internal { rule: NAMES_A_ROW }),
    }
}

/// The key form `form` at each of `len` elements of `tensor`, `at` giving a term's value at an
/// element: the row it names, below `rows`.
fn keys_of<T: Copy>(
    tensor: TensorId,
    form: &Affine<Goldilocks, T>,
    len: usize,
    rows: u64,
    at: &impl Fn(T, usize) -> Goldilocks,
) -> Result<SecretBuf<u32>, WitnessError> {
    let mut keys =
        SecretBuf::try_with_capacity(len).map_err(|_| WitnessError::Memory { tensor })?;
    let mut fits = true;
    for i in 0..len {
        let key = form
            .evaluate::<Goldilocks>(|term| at(term, i))
            .to_canonical_u64();
        let named = u32::try_from(key).ok().filter(|&k| u64::from(k) < rows);
        fits &= named.is_some();
        keys.push(named.unwrap_or(0));
    }
    match fits {
        true => Ok(keys),
        false => Err(WitnessError::Internal { rule: NAMES_A_ROW }),
    }
}

/// The tile's committed tensor that holds a lookup's values: the value form is one committed
/// tensor, `y` or one of the gadget's.
fn value_column(
    form: &Affine<Goldilocks>,
    y: usize,
    gadget: &[(AuxColumn, SecretBuf<i64>)],
) -> Result<Column, WitnessError> {
    let column = match (form.constant == Goldilocks::ZERO, form.terms.as_slice()) {
        (true, &[(one, Term::Committed(Committed::Y))]) if one == Goldilocks::ONE => {
            Some(Column::Share(y))
        }
        (true, &[(one, Term::Committed(c))]) if one == Goldilocks::ONE => gadget
            .iter()
            .position(|(g, _)| requant_role(c).is_ok_and(|role| *g == role))
            .map(Column::Gadget),
        _ => None,
    };
    column.ok_or(WitnessError::Internal {
        rule: "a lookup's value is one committed tensor of its tile",
    })
}

impl Gadgets {
    /// Selected requantization tables are not graph tables held by the engine.
    pub(crate) fn rows(
        &self,
        structure: &Structure<'_>,
        table: TableId,
    ) -> Result<u64, WitnessError> {
        if self.incomplete.contains(&table) {
            return usage(
                "future greedy and sampling multiplicities are outside the current model layout",
            );
        }
        match self
            .requant
            .iter()
            .flatten()
            .flat_map(|g| &g.rows)
            .chain(self.auxiliary.iter().flatten().flat_map(Prepared::rows))
            .find(|(id, _)| *id == table)
        {
            Some(&(_, rows)) => Ok(rows),
            None => Ok(structure.engine.lookup(table)?.row_count()),
        }
    }

    /// Whether a node contributes any lookup to this table.
    pub(crate) fn reads_node(&self, node: &Node, table: TableId) -> bool {
        self.requests[node.id.0 as usize].contains(&table)
    }

    /// Public upper bound on insertions, including weighted padding and the row-0 tail.
    pub(crate) fn update_bound(
        &self,
        structure: &Structure<'_>,
        table: TableId,
    ) -> Result<u64, WitnessError> {
        let error = WitnessError::MemoryTable { table };
        let graph = structure.graph();
        let mut total = 1u64;
        for node in &graph.nodes {
            let frequency = self.requests[node.id.0 as usize]
                .iter()
                .filter(|&&id| id == table)
                .count() as u64;
            if frequency == 0 {
                continue;
            }
            let tensor = match node.op {
                Op::SoftmaxRow { exp_table, .. } if exp_table != table => node.outputs[1],
                _ => node.inputs[0],
            };
            let shape = &graph.tensors[tensor.0 as usize].shape;
            let real = shape
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(u64::from(d)))
                .ok_or(error)?;
            let padded = shape
                .iter()
                .try_fold(1u64, |n, &d| {
                    n.checked_mul(u64::from(d).next_power_of_two())
                })
                .ok_or(error)?;
            let padding_calls = if real == padded {
                0
            } else if matches!(node.op, Op::Requant { .. } | Op::TableMap { .. })
                && node.inputs.len() > 1
            {
                u64::from(*shape.last().ok_or(error)?)
                    .checked_add(1)
                    .ok_or(error)?
            } else {
                1
            };
            total = frequency
                .checked_mul(real.checked_add(padding_calls).ok_or(error)?)
                .and_then(|n| total.checked_add(n))
                .ok_or(error)?;
        }
        Ok(total)
    }

    /// Per-dimension padding is a lookup of the node's zero inputs, with real channel classes
    /// still broadcast on padded token rows (RFC-0035, RFC-0045); masked Exp keys use K_0.
    pub(crate) fn padding(
        &self,
        structure: &Structure<'_>,
        node: &Node,
        counts: &mut Multiplicities,
        read_classes: impl FnOnce(TensorId) -> Result<SecretBuf<i64>, WitnessError>,
    ) -> Result<(), WitnessError> {
        let input = node.inputs[0];
        let domain = match node.op {
            Op::SoftmaxRow { exp_table, .. } if exp_table != counts.table => node.outputs[1],
            _ => input,
        };
        let shape = &structure.graph().tensors[domain.0 as usize].shape;
        let product = |padded: bool| {
            shape.iter().try_fold(1u64, |size, &d| {
                size.checked_mul(if padded {
                    u64::from(d).next_power_of_two()
                } else {
                    u64::from(d)
                })
                .ok_or(WitnessError::Internal {
                    rule: "a lookup domain fits u64",
                })
            })
        };
        let (real, padded) = (product(false)?, product(true)?);
        if padded == real {
            return Ok(());
        }
        if let Op::SoftmaxRow { exp_table, .. } = node.op {
            if counts.table != exp_table {
                for &table in &self.requests[node.id.0 as usize] {
                    if table == counts.table {
                        counts.add(0, padded - real)?;
                    }
                }
                return Ok(());
            }
            let table = structure.engine.lookup(exp_table)?;
            let TableKind::Exp { key_bits, .. } = table.spec().kind else {
                return Err(WitnessError::Internal {
                    rule: "SoftmaxRow reads an Exp table",
                });
            };
            return counts.add((1u64 << key_bits) - 1, padded - real);
        }
        if matches!(node.op, Op::RsqrtRow { .. }) {
            for &table in &self.requests[node.id.0 as usize] {
                if table == counts.table {
                    counts.add(0, padded - real)?;
                }
            }
            return Ok(());
        }
        let classes = if matches!(node.op, Op::Requant { .. } | Op::TableMap { .. }) {
            node.inputs.get(1).copied().map(read_classes).transpose()?
        } else {
            None
        };
        let Some(classes) = classes else {
            return self.zero_lookups(structure, node, 0, padded - real, counts);
        };
        let width = u64::from(*shape.last().ok_or(WitnessError::Internal {
            rule: "a lookup has an axis",
        })?);
        if classes.len() != width as usize {
            return Err(WitnessError::Internal {
                rule: "a channel class vector has one value per channel",
            });
        }
        let padded_rows = padded / width.next_power_of_two();
        let extra_rows = padded_rows - real / width;
        if extra_rows > 0 {
            for c in 0..width as usize {
                let value = classes[c];
                let class = u8::try_from(value).map_err(|_| WitnessError::Internal {
                    rule: "a channel class fits u8",
                })?;
                self.zero_lookups(structure, node, class, extra_rows, counts)?;
            }
        }
        let extra = (width.next_power_of_two() - width) * padded_rows;
        if extra > 0 {
            self.zero_lookups(structure, node, 0, extra, counts)?;
        }
        Ok(())
    }

    fn zero_lookups(
        &self,
        structure: &Structure<'_>,
        node: &Node,
        class: u8,
        copies: u64,
        counts: &mut Multiplicities,
    ) -> Result<(), WitnessError> {
        if let Op::Requant { .. } = node.op {
            let gadget =
                self.requant[node.id.0 as usize]
                    .as_ref()
                    .ok_or(WitnessError::Internal {
                        rule: "the selection gives every requantization its gadget",
                    })?;
            let witness = gadget.requant.witness(&[0], Some(&[class]))?;
            let table = counts.table;
            for lookup in gadget
                .constraints
                .lookups
                .iter()
                .filter(|l| l.table == table)
            {
                let mut complete = true;
                let key = lookup.key.evaluate::<Goldilocks>(|term| match term {
                    Term::Input => Goldilocks::ZERO,
                    Term::Class => Goldilocks::from_u64_reduced(u64::from(class)),
                    Term::Committed(c) => match witness.tensor(c).and_then(|v| v.first()) {
                        Some(&v) => Goldilocks::from_i64(v),
                        None => {
                            complete = false;
                            Goldilocks::ZERO
                        }
                    },
                });
                if !complete {
                    return Err(WitnessError::Internal {
                        rule: "a padding lookup reads its gadget's values",
                    });
                }
                counts.add(key.to_canonical_u64(), copies)?;
            }
            Ok(())
        } else {
            let table = structure.engine.lookup(counts.table)?;
            let key = table
                .index(0, class)
                .ok_or(WitnessError::Internal { rule: NAMES_A_ROW })?;
            counts.add(key, copies)
        }
    }
}
