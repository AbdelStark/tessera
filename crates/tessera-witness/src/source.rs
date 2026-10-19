// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The witness sources (RFC-0018 'Pass structure', RFC-0010 'Stages'): any tile's values on
//! demand, by tile (RFC-0041 'Tiles').
//!
//! - [`Regenerator`] regenerates a tile from the checkpoints: it replays the tile's layer from its
//!   checkpoint up to the tile's node, the cone of that node alone, and holds nothing between
//!   tiles. Its memory is the checkpoints and the tile it regenerates, with the replay's tensors,
//!   none of which outgrows a tile: an attention tile computes its own rows of the scores, and the
//!   tiles after a layer's attention run it row by row.
//! - [`Resident`] cuts tiles out of the first full pass's witness, every tensor in memory.
//!
//! Both give the same values for every tile, the first pass's, bit for bit: the regeneration runs
//! the reference engine's own operations on the same inputs.

use core::fmt;

use tessera_codec::{Secret, SecretBuf};
use tessera_compiler::{QueryBlock, Selection, SequenceTile, Tile};
use tessera_ir::{IntegerModel, TableId};

use crate::engine::{Input, Operand, elements};
use crate::replay::{Replay, Rows};
use crate::structure::Structure;
use crate::tile::{Gadgets, Multiplicities, Target};
use crate::{Checkpoints, Engine, TileWitness, Witness, WitnessError};

/// A source of a sequence's witness, by tile (RFC-0018 'Pass structure', RFC-0010 'Stages'): the
/// passes after the first regenerate the parts of the witness they need, tile by tile, instead of
/// storing it.
pub trait WitnessSource {
    /// The values of `tile`, a tile of the source's sequence as `ExecutionPlan::expand` gives it
    /// (RFC-0041 'Tiles'): its shares' elements, its gadget's committed values and its lookups'
    /// keys and values ([`TileWitness`]).
    ///
    /// # Errors
    ///
    /// Component tiles do not yet include all SoftmaxRow/RsqrtRow/Argmax gadget data.
    ///
    /// [`WitnessError::Usage`] for an auxiliary SoftmaxRow/Argmax table whose full counts are not
    /// implemented, or a tile that is not one of the sequence's, with the shares
    /// RFC-0041 gives it; otherwise the source's own.
    fn tile(&self, tile: &SequenceTile) -> Result<TileWitness, WitnessError>;
}

/// Regenerates any tile of a proof unit's sequence from its [`Checkpoints`] (RFC-0018 'Witness
/// engine'): the tile's layer, from its checkpoint, up to its node, through the reference engine's
/// operations, so every value is the first full pass's.
///
/// It borrows the engine, the model, the tokens and the checkpoints, and holds nothing else between
/// tiles but the public gadgets. The tokens are secret-class (05 §3): the engine reads them, and
/// every value regenerated lives in a `SecretBuf`.
pub struct Regenerator<'a> {
    replay: Replay<'a>,
    gadgets: Gadgets,
    store: &'a Checkpoints,
}

/// Shows the shapes: the graph's tensors and the checkpoints.
impl fmt::Debug for Regenerator<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Regenerator")
            .field("tensors", &self.replay.structure.graph().tensors.len())
            .field("checkpoints", &self.store.len())
            .finish_non_exhaustive()
    }
}

impl<'a> Regenerator<'a> {
    /// The regenerator of `model` on `tokens`, on `engine`, the engine of `model`'s lowering for
    /// the tokens' positions, with the gadgets `selection` gives that lowering's nodes and the
    /// store `checkpoints` holds for the same model and tokens.
    ///
    /// # Errors
    ///
    /// - as [`Engine::run`] for the engine, the tokens and the model, in its order;
    /// - [`WitnessError::Usage`] for checkpoints of another lowering or sequence length;
    /// - [`WitnessError::Internal`] and [`WitnessError::Gadget`] for a selection without a
    ///   requantization's gadget, or a gadget that refuses its tables.
    pub fn new(
        engine: &'a Engine<'a>,
        selection: &Selection,
        model: &'a IntegerModel,
        tokens: &'a Secret<Vec<u32>>,
        checkpoints: &'a Checkpoints,
    ) -> Result<Self, WitnessError> {
        let structure = Structure::new(engine, &model.arch)?;
        let replay = Replay::new(structure, model, tokens)?;
        if !checkpoints.fits(replay.lowering()) {
            return Err(WitnessError::Usage {
                rule: "the checkpoints are the lowering's, for its sequence",
            });
        }
        let gadgets = Gadgets::new(&replay.structure, selection)?;
        Ok(Self {
            replay,
            gadgets,
            store: checkpoints,
        })
    }
}

impl WitnessSource for Regenerator<'_> {
    /// Regenerates `tile` from the checkpoints: the tile's node's cone, back to its layer's
    /// checkpoint and the graph's inputs, then its node, or an attention tile's own rows.
    ///
    /// # Errors
    ///
    /// [`WitnessError::Usage`] for a tile that is not one of the sequence's; as [`Engine::run`]
    /// otherwise: a value outside its range in debug builds, a buffer that cannot be allocated, or
    /// a key outside its table (`ERR-INTERNAL`).
    fn tile(&self, tile: &SequenceTile) -> Result<TileWitness, WitnessError> {
        let structure = &self.replay.structure;
        match structure.target(tile, &self.gadgets.read)? {
            Target::Tokens => Ok(TileWitness::tokens(self.replay.tokens()?)),
            Target::Table(table) => {
                multiplicities(self, structure, &self.gadgets, table, |tensor| {
                    match self.replay.leaf(self.store, tensor)? {
                        Input::Values(values) => Ok(values),
                        Input::Weight(_) => Err(WitnessError::Internal {
                            rule: "channel classes are integer vectors",
                        }),
                    }
                })
            }
            Target::Node(id) => {
                let node = &structure.graph().nodes[id as usize];
                let live = self.replay.compute(self.store, &node.inputs)?;
                let operands = self.replay.operands(self.store, node, &live)?;
                let inputs = operands.all(&live)?;
                let outputs = self.replay.run(node, &inputs)?;
                self.gadgets
                    .node_tile(structure, node, &inputs, outputs, &tile.shares)
            }
            Target::Attention {
                core,
                heads,
                queries,
            } => {
                let core = &structure.cores[core];
                // A block of padding alone holds no element: nothing to replay.
                let rows = if queries.is_empty() {
                    Rows {
                        scores: SecretBuf::new(),
                        e: SecretBuf::new(),
                        r: SecretBuf::new(),
                    }
                } else {
                    let live = self.replay.compute(self.store, &[core.q, core.k])?;
                    self.replay
                        .attention_rows(core, &live, heads, queries.clone())?
                };
                self.gadgets.attention_tile(structure, core, rows, queries)
            }
        }
    }
}

/// Cuts tiles out of the first full pass's [`Witness`], every tensor in memory: the source of a
/// pass that keeps the witness, and the reference a regeneration equals.
pub struct Resident<'a> {
    structure: Structure<'a>,
    gadgets: Gadgets,
    witness: &'a Witness,
}

/// Shows the shapes: the graph's tensors.
impl fmt::Debug for Resident<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Resident")
            .field("tensors", &self.structure.graph().tensors.len())
            .finish_non_exhaustive()
    }
}

impl<'a> Resident<'a> {
    /// The source of `witness`, [`Engine::run`]'s on `engine` for `model`, with the gadgets
    /// `selection` gives the lowering's nodes.
    ///
    /// # Errors
    ///
    /// - [`WitnessError::Usage`] for an engine made without a lowering, or a witness that is not a
    ///   run of its graph: another number of tensors, or a tensor of another size;
    /// - [`WitnessError::Model`] for an architecture whose KV heads do not divide its query heads;
    /// - [`WitnessError::Internal`] and [`WitnessError::Gadget`] as [`Regenerator::new`].
    pub fn new(
        engine: &'a Engine<'a>,
        selection: &Selection,
        model: &IntegerModel,
        witness: &'a Witness,
    ) -> Result<Self, WitnessError> {
        let structure = Structure::new(engine, &model.arch)?;
        let graph = structure.graph();
        let whole = witness.len() == graph.tensors.len()
            && graph.tensors.iter().enumerate().all(|(t, spec)| {
                witness
                    .tensor(tessera_ir::TensorId(t as u32))
                    .is_some_and(|values| values.len() == elements(&spec.shape))
            });
        if !whole {
            return Err(WitnessError::Usage {
                rule: "the witness is the first pass of the engine's lowering",
            });
        }
        let gadgets = Gadgets::new(&structure, selection)?;
        Ok(Self {
            structure,
            gadgets,
            witness,
        })
    }

    fn values(&self, tensor: tessera_ir::TensorId) -> Result<&'a [i64], WitnessError> {
        self.witness
            .tensor(tensor)
            .map(|values| &values[..])
            .ok_or(WitnessError::Internal {
                rule: "the first pass holds every tensor",
            })
    }

    /// `values`, copied into a buffer of its own.
    fn copied(
        tensor: tessera_ir::TensorId,
        values: &[i64],
    ) -> Result<SecretBuf<i64>, WitnessError> {
        let mut buffer = SecretBuf::try_with_capacity(values.len())
            .map_err(|_| WitnessError::Memory { tensor })?;
        buffer.extend(values.iter().copied());
        Ok(buffer)
    }
}

impl WitnessSource for Resident<'_> {
    /// Cuts `tile` out of the first pass's tensors.
    ///
    /// # Errors
    ///
    /// [`WitnessError::Usage`] for a tile that is not one of the sequence's;
    /// [`WitnessError::Memory`] when a buffer cannot be allocated.
    fn tile(&self, tile: &SequenceTile) -> Result<TileWitness, WitnessError> {
        let structure = &self.structure;
        match structure.target(tile, &self.gadgets.read)? {
            Target::Tokens => {
                let tokens = structure.lowering.tokens;
                Ok(TileWitness::tokens(Self::copied(
                    tokens,
                    self.values(tokens)?,
                )?))
            }
            Target::Table(table) => {
                multiplicities(self, structure, &self.gadgets, table, |tensor| {
                    Self::copied(tensor, self.values(tensor)?)
                })
            }
            Target::Node(id) => {
                let node = &structure.graph().nodes[id as usize];
                let inputs = node
                    .inputs
                    .iter()
                    .map(|&t| self.values(t).map(Operand::Values))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = node
                    .outputs
                    .iter()
                    .map(|&t| Self::copied(t, self.values(t)?))
                    .collect::<Result<Vec<_>, _>>()?;
                self.gadgets
                    .node_tile(structure, node, &inputs, outputs, &tile.shares)
            }
            Target::Attention {
                core,
                heads,
                queries,
            } => {
                let core = &structure.cores[core];
                let (scores, e, r) = (
                    self.values(core.scores)?,
                    self.values(core.e)?,
                    self.values(core.r)?,
                );
                let n = core.n;
                let rows = heads.len() * queries.len();
                let mut cut = Rows {
                    scores: SecretBuf::try_with_capacity(rows * n).map_err(|_| {
                        WitnessError::Memory {
                            tensor: core.scores,
                        }
                    })?,
                    e: SecretBuf::try_with_capacity(rows * n)
                        .map_err(|_| WitnessError::Memory { tensor: core.e })?,
                    r: SecretBuf::try_with_capacity(rows)
                        .map_err(|_| WitnessError::Memory { tensor: core.r })?,
                };
                for head in heads {
                    for query in queries.clone() {
                        let row = head * n + query;
                        cut.scores
                            .extend(scores[row * n..(row + 1) * n].iter().copied());
                        cut.e.extend(e[row * n..(row + 1) * n].iter().copied());
                        cut.r.push(r[row]);
                    }
                }
                self.gadgets.attention_tile(structure, core, cut, queries)
            }
        }
    }
}

/// Recount a table from contributing nodes, at most one query row at a time. No other table's
/// multiplicities or tile survives a visit; the column's row-0 tail is added after all blocks.
fn multiplicities(
    source: &impl WitnessSource,
    structure: &Structure<'_>,
    gadgets: &Gadgets,
    table: TableId,
    read_classes: impl Fn(tessera_ir::TensorId) -> Result<SecretBuf<i64>, WitnessError>,
) -> Result<TileWitness, WitnessError> {
    let rows = gadgets.rows(structure, table)?;
    let mut counts = Multiplicities::new(table, rows)?;
    let visit = |tile: SequenceTile, counts: &mut Multiplicities| {
        let witness = source.tile(&tile)?;
        for lookup in witness.lookups().filter(|lookup| lookup.table() == table) {
            for &key in lookup.keys() {
                counts.add(u64::from(key), 1)?;
            }
        }
        Ok::<(), WitnessError>(())
    };
    for node in &structure.graph().nodes {
        if !gadgets.reads_node(node, table) {
            continue;
        }
        if let Some(core) = structure.core_of(node.id.0) {
            let n = structure.cores[core].n as u32;
            for kv_head in 0..structure.kv_heads() {
                let heads = structure
                    .query_heads(kv_head)
                    .ok_or(WitnessError::Internal {
                        rule: "a KV head has query heads",
                    })?;
                for query in 0..n {
                    let mut shares: Vec<_> = node
                        .outputs
                        .iter()
                        .filter_map(|&t| structure.whole(t))
                        .collect();
                    for share in &mut shares {
                        share.ranges[0] = heads.start as u32..heads.end as u32;
                        share.ranges[1] = query..query + 1;
                    }
                    visit(
                        SequenceTile {
                            tile: Tile::Attention {
                                node: node.id.0,
                                kv_head,
                            },
                            block: Some(QueryBlock {
                                index: query,
                                queries: query..query + 1,
                            }),
                            shares,
                        },
                        &mut counts,
                    )?;
                }
            }
        } else {
            visit(
                SequenceTile {
                    tile: Tile::Node { node: node.id.0 },
                    block: None,
                    shares: node
                        .outputs
                        .iter()
                        .filter_map(|&t| structure.whole(t))
                        .collect(),
                },
                &mut counts,
            )?;
        }
        gadgets.padding(structure, node, &mut counts, &read_classes)?;
    }
    counts.finish()
}
