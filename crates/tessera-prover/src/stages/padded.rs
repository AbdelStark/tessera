// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Padded positions (RFC-0072 'Padded positions'): what a tensor holds at the positions its
//! per-dimension padding adds (RFC-0011 rule 1), which the stage tables and the leaf stage's
//! blocks read.
//!
//! Every tensor's padded positions hold what its operation computes from its inputs' padded
//! positions (RFC-0035 'What a padded row holds'). Graph inputs and constants pad with zeros, and
//! every operation maps zeros there to zeros but a lookup whose function does not, such as the
//! S-box, `AesSbox(0) = 0x63`. So a padded position is non-zero only downstream of a lookup node
//! non-zero at padding, which the compiler decides from public data
//! ([`tensors_nonzero_at_padding`]). [`PaddedWitness`] computes those tensors over their padded
//! shapes; every other tensor is its values padded with zeros.
//!
//! **Why here.** The CPU engine is the bit-exact oracle of `D` over real shapes, which the GPU
//! engine and the PyTorch reference reproduce (07 §5); padded positions belong to the proof's
//! layout, not to `D`. The stage tables ([`WitnessSource::padded`]), the leaf stage's blocks
//! ([`GraphLeafWitness`](crate::leaf::GraphLeafWitness)) and the commitment all read one
//! [`PaddedWitness`], so they cannot disagree, and a graph with no such lookup, every Llama
//! graph, computes nothing here.

use core::fmt;
use std::collections::HashMap;

use tessera_codec::SecretBuf;
use tessera_compiler::CompileError;
use tessera_compiler::schedule::{reads_padding, tensors_nonzero_at_padding};
use tessera_ir::{Broadcast, Graph, Node, Op, OpClass, TensorId};
use tessera_tables::{GeneratedTable, RopeTable, Table, TableId, TableKind, TableSpec};

use super::{StageError, WitnessSource};

/// A witness with the padded positions of every tensor a lookup non-zero at padding reaches
/// (RFC-0072 'Padded positions'): each such tensor over its padded shape, flat and row-major,
/// the last axis's index the lowest bits ([`WitnessSource::padded`]). Its real values are its
/// source's.
///
/// The tables are secret-class (05 §3): they live in `SecretBuf`s that clear their memory, and
/// the debug form shows how many there are, never a value.
pub struct PaddedWitness<'a, W: ?Sized> {
    witness: &'a W,
    tables: Vec<Option<SecretBuf<i64>>>,
}

/// Shows the count of padded tables, never a value (05 §3).
impl<W: ?Sized> fmt::Debug for PaddedWitness<'_, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PaddedWitness")
            .field("padded", &self.tables.iter().flatten().count())
            .finish_non_exhaustive()
    }
}

impl<'a, W: WitnessSource + ?Sized> PaddedWitness<'a, W> {
    /// The padded tables of `graph`, whose lookups read the tables `tables`, over the real values
    /// of `witness`: for each tensor that [`tensors_nonzero_at_padding`] marks, in node order,
    /// what its operation computes at each position of its padded shape from its inputs' padded
    /// tables, an input it does not mark being its values padded with zeros:
    ///
    /// - **a lookup**, `TableMap`, `Bitwise` or `Requant`, looks up what its key packs there,
    ///   such as `AesSbox(0) = 0x63`, a padded channel's class being 0 (RFC-0045 'Blocks');
    /// - **a `ConstAdd`** adds its constant at real positions only, and a `ConstMul` by a
    ///   constant per channel multiplies a padded channel by 0;
    /// - **a selection** is 0 on its padded rows along its axis, **an `Expand`** past its length,
    ///   and **a `Rotate`** on its padded rows; their other padded positions are their input's;
    /// - **a sum**, `Sum`, `SumSquares` or a matmul, runs over its axis's whole padded cube, as
    ///   its relation does (RFC-0010 'Relations');
    /// - **a `BitDecompose`** gives the bits of its input's value there, 0 in its padded bit
    ///   slots; **`BitFn`**, `Add`, `Sub` and `Mul` work position by position;
    /// - **a `OneHot`, `SoftmaxRow` or `RsqrtRow`** whose input has no padded position holds its
    ///   values padded with zeros, which RFC-0035's padding checks pin.
    ///
    /// At its real positions each table must be its source's values: a sum over a padded axis
    /// that RFC-0072 'Summing relations' refuses is the one way to break that.
    ///
    /// # Errors
    ///
    /// [`StageError::Internal`] (`ERR-INTERNAL`) for the first rule broken, node by node: a lookup
    /// that the compiler cannot classify or whose table does not regenerate, a value or a key no
    /// table or integer holds, a `Reshape` that does not map its padded cube as the identity, a
    /// `Rotate` whose pairs are not a power of two, a gadget other than a lookup, `BitDecompose`
    /// or `AssertEqual` reading such a tensor with padded positions, whose row relations take a
    /// padded row's input as 0 (RFC-0035; the schedule refuses it too), a tensor `witness` lacks,
    /// or a table that is not `witness`'s values at its real positions. [`StageError::Memory`]
    /// when a table does not fit.
    pub fn new(graph: &Graph, tables: &[TableSpec], witness: &'a W) -> Result<Self, StageError> {
        let reached = tensors_nonzero_at_padding(graph, tables).map_err(|error| match error {
            CompileError::Schedule { rule, .. } => StageError::Internal { rule },
            _ => StageError::Internal {
                rule: "a lookup's padding is decided from public data",
            },
        })?;
        let marked = |t: &TensorId| reached.get(t.0 as usize).copied().unwrap_or(false);
        let mut padded: Vec<Option<SecretBuf<i64>>> =
            (0..graph.tensors.len()).map(|_| None).collect();
        let mut evaluator = Evaluator {
            graph,
            tables,
            lookups: HashMap::new(),
        };
        let padded_input = |t: &TensorId| {
            marked(t)
                && graph
                    .tensors
                    .get(t.0 as usize)
                    .is_some_and(|spec| spec.shape.iter().any(|len| !len.is_power_of_two()))
        };
        for node in &graph.nodes {
            // The schedule's rule, the same (`derive_graph_schedule`'s errors).
            let gadget = node.op.class() == OpClass::Gadget;
            if gadget && !reads_padding(&node.op) && node.inputs.iter().any(padded_input) {
                return internal(
                    "no gadget but a lookup, BitDecompose or AssertEqual reads a tensor non-zero \
                     at its padded positions: RFC-0035's row relations take a padded row's input \
                     as 0",
                );
            }
            if !node.outputs.iter().any(marked) {
                continue;
            }
            let outputs = {
                let mut owned = Vec::with_capacity(node.inputs.len());
                for &t in &node.inputs {
                    owned.push(match padded.get(t.0 as usize).and_then(Option::as_ref) {
                        Some(_) => None,
                        None => Some(zero_padded(graph, witness, t)?),
                    });
                }
                let inputs: Vec<&[i64]> = node
                    .inputs
                    .iter()
                    .zip(&owned)
                    .map(|(t, own)| match own {
                        Some(table) => &table[..],
                        None => padded[t.0 as usize].as_deref().unwrap_or(&[]),
                    })
                    .collect();
                evaluator.node(node, &inputs, witness)?
            };
            if outputs.len() != node.outputs.len() {
                return internal("a node's padded outputs are its outputs");
            }
            for (&t, table) in node.outputs.iter().zip(outputs) {
                let values = witness.values(t).ok_or(StageError::Internal {
                    rule: "the witness holds every tensor a lookup non-zero at padding reaches",
                })?;
                if !agrees(&shape(graph, t)?, &table, values) {
                    return internal(
                        "a padded table is the witness's values at its real positions (RFC-0072 \
                         'Summing relations')",
                    );
                }
                padded[t.0 as usize] = Some(table);
            }
        }
        Ok(Self {
            witness,
            tables: padded,
        })
    }
}

/// Real values from the source, padded positions from the tables computed for the tensors a
/// lookup non-zero at padding reaches, else the source's own.
impl<W: WitnessSource + ?Sized> WitnessSource for PaddedWitness<'_, W> {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        self.witness.values(tensor)
    }

    fn int8_values(&self, tensor: TensorId) -> Option<&[i8]> {
        self.witness.int8_values(tensor)
    }

    fn padded(&self, tensor: TensorId) -> Option<&[i64]> {
        match self.tables.get(tensor.0 as usize) {
            Some(Some(table)) => Some(table),
            _ => self.witness.padded(tensor),
        }
    }
}

/// A shape over its padded cube: each axis's length rounded up to a power of two, row-major, the
/// last axis taking the lowest bits of an index (RFC-0011 rule 1).
struct Cube {
    dims: Vec<usize>,
    bits: Vec<u32>,
    low: Vec<u32>,
}

impl Cube {
    fn of(shape: &[u32]) -> Self {
        let dims: Vec<usize> = shape.iter().map(|&len| len as usize).collect();
        let bits: Vec<u32> = shape
            .iter()
            .map(|&len| len.next_power_of_two().trailing_zeros())
            .collect();
        let mut low = vec![0; shape.len()];
        let mut at = 0;
        for axis in (0..shape.len()).rev() {
            low[axis] = at;
            at += bits[axis];
        }
        Self { dims, bits, low }
    }

    /// `2^n`, the padded cube's positions.
    fn len(&self) -> usize {
        1usize << self.bits.iter().sum::<u32>()
    }

    /// The bits of the last axis.
    fn last_bits(&self) -> u32 {
        self.bits.last().copied().unwrap_or(0)
    }

    /// The coordinate of `axis` at padded index `index`.
    fn coord(&self, index: usize, axis: usize) -> usize {
        (index >> self.low[axis]) & ((1usize << self.bits[axis]) - 1)
    }

    /// The padded index of the coordinates `coords`, one per axis.
    fn index(&self, coords: &[usize]) -> usize {
        coords
            .iter()
            .zip(&self.low)
            .fold(0, |index, (&c, &low)| index | (c << low))
    }

    /// The coordinates of padded index `index`.
    fn coords(&self, index: usize) -> Vec<usize> {
        (0..self.dims.len()).map(|a| self.coord(index, a)).collect()
    }

    /// Whether padded index `index` is a real position: inside the shape on every axis.
    fn real(&self, index: usize) -> bool {
        (0..self.dims.len()).all(|a| self.coord(index, a) < self.dims[a])
    }

    /// The padded index of the real position `flat`, row-major over the shape.
    fn of_flat(&self, flat: usize) -> usize {
        let mut rest = flat;
        let mut index = 0;
        for axis in (0..self.dims.len()).rev() {
            let len = self.dims[axis].max(1);
            index |= (rest % len) << self.low[axis];
            rest /= len;
        }
        index
    }

    /// The real positions, `Π n_a`.
    fn count(&self) -> usize {
        self.dims.iter().product()
    }
}

/// The shape of `tensor`.
fn shape(graph: &Graph, tensor: TensorId) -> Result<Vec<u32>, StageError> {
    graph
        .tensors
        .get(tensor.0 as usize)
        .map(|spec| spec.shape.clone())
        .ok_or(StageError::Internal {
            rule: "a tensor is the graph's",
        })
}

/// `tensor`'s values padded per dimension with zeros, in a buffer that clears its memory.
fn zero_padded<W: WitnessSource + ?Sized>(
    graph: &Graph,
    witness: &W,
    tensor: TensorId,
) -> Result<SecretBuf<i64>, StageError> {
    let cube = Cube::of(&shape(graph, tensor)?);
    let values = super::tables::Integers::real(witness, tensor)?;
    if values.len() != cube.count() {
        return internal("a tensor holds its shape's number of values");
    }
    let mut table = zeros(cube.len())?;
    for flat in 0..values.len() {
        table[cube.of_flat(flat)] = values.at(flat);
    }
    Ok(table)
}

/// Whether `table`, over the padded shape of `shape`, holds `values` at its real positions.
fn agrees(shape: &[u32], table: &[i64], values: &[i64]) -> bool {
    let cube = Cube::of(shape);
    table.len() == cube.len()
        && values.len() == cube.count()
        && values
            .iter()
            .enumerate()
            .all(|(flat, &v)| table[cube.of_flat(flat)] == v)
}

/// A table of `len` zeros that clears its memory.
fn zeros(len: usize) -> Result<SecretBuf<i64>, StageError> {
    let mut table = SecretBuf::try_with_capacity(len).map_err(|_| StageError::Memory)?;
    table.resize(len, 0);
    Ok(table)
}

/// `value` as an `i64`, or the overflow refused.
fn fit(value: i128) -> Result<i64, StageError> {
    i64::try_from(value).map_err(|_| StageError::Internal {
        rule: "a padded value is a true integer of 64 bits",
    })
}

/// The padded evaluation of a graph's nodes, with the lookup tables it has regenerated.
struct Evaluator<'g> {
    graph: &'g Graph,
    tables: &'g [TableSpec],
    lookups: HashMap<TableId, GeneratedTable>,
}

impl Evaluator<'_> {
    /// Lookup table `id`, regenerated from its specification once.
    fn lookup(&mut self, id: TableId) -> Result<&GeneratedTable, StageError> {
        if !self.lookups.contains_key(&id) {
            let spec =
                self.tables
                    .iter()
                    .find(|spec| spec.id == id)
                    .ok_or(StageError::Internal {
                        rule: "a lookup's table is the graph's",
                    })?;
            let table = GeneratedTable::from_spec(spec).map_err(|error| {
                StageError::from_table(error, "a lookup's table regenerates from its specification")
            })?;
            self.lookups.insert(id, table);
        }
        self.lookups.get(&id).ok_or(StageError::Internal {
            rule: "a lookup's table is the graph's",
        })
    }

    /// The constant `c` of the graph.
    fn constant(&self, c: tessera_ir::ConstRef) -> Result<&[i64], StageError> {
        self.graph
            .constants
            .get(c.0 as usize)
            .map(Vec::as_slice)
            .ok_or(StageError::Internal {
                rule: "a constant the graph holds",
            })
    }

    /// `node`'s outputs over their padded shapes from its inputs' padded tables `inputs`.
    fn node<W: WitnessSource + ?Sized>(
        &mut self,
        node: &Node,
        inputs: &[&[i64]],
        witness: &W,
    ) -> Result<Vec<SecretBuf<i64>>, StageError> {
        let graph = self.graph;
        let input = |k: usize| -> Result<&[i64], StageError> {
            inputs.get(k).copied().ok_or(StageError::Internal {
                rule: "a node's inputs are computed before it",
            })
        };
        let cube_in = |k: usize| -> Result<Cube, StageError> {
            let t = node.inputs.get(k).ok_or(StageError::Internal {
                rule: "a node has its inputs",
            })?;
            Ok(Cube::of(&shape(graph, *t)?))
        };
        let out = Cube::of(&shape(
            graph,
            *node.outputs.first().ok_or(StageError::Internal {
                rule: "a node with a padded table has an output",
            })?,
        )?);
        let mut y = zeros(out.len())?;
        match &node.op {
            Op::Add | Op::Sub => {
                let (a, b) = (input(0)?, input(1)?);
                let sub = node.op == Op::Sub;
                for (i, slot) in y.iter_mut().enumerate() {
                    let (p, q) = (i128::from(a[i]), i128::from(b[i]));
                    *slot = fit(if sub { p - q } else { p + q })?;
                }
            }
            Op::Mul { broadcast } => {
                let (a, b) = (input(0)?, input(1)?);
                let (a_bits, b_bits) = (
                    cube_in(0)?.len().trailing_zeros(),
                    cube_in(1)?.len().trailing_zeros(),
                );
                for (i, slot) in y.iter_mut().enumerate() {
                    // `b` is `a`'s leading axes, its highest bits, or its last axis, its lowest.
                    let j = match broadcast {
                        Broadcast::Trailing => i >> (a_bits - b_bits),
                        Broadcast::Leading => i & ((1usize << b_bits) - 1),
                    };
                    *slot = fit(i128::from(a[i]) * i128::from(b[j]))?;
                }
            }
            Op::ConstMul { c } | Op::ConstAdd { c } => {
                let x = input(0)?;
                let constant = self.constant(*c)?;
                let channels = out.dims.last().copied().unwrap_or(1);
                for (i, slot) in y.iter_mut().enumerate() {
                    // A constant per channel is 0 at padded channels.
                    let channel = i & ((1usize << out.last_bits()) - 1);
                    let k = match constant {
                        [scalar] => *scalar,
                        _ if channel < channels => constant[channel],
                        _ => 0,
                    };
                    *slot = fit(if matches!(node.op, Op::ConstMul { .. }) {
                        i128::from(x[i]) * i128::from(k)
                    } else if out.real(i) {
                        // At real positions only (RFC-0072 'Padded positions').
                        i128::from(x[i]) + i128::from(k)
                    } else {
                        i128::from(x[i])
                    })?;
                }
            }
            Op::Reshape => {
                // The schedule maps a claim through a `Reshape` only when it maps the padded
                // cubes' coordinates as the identity (RFC-0037 'Head layouts').
                let x = cube_in(0)?;
                if x.len() != out.len()
                    || (0..out.count()).any(|flat| x.of_flat(flat) != out.of_flat(flat))
                {
                    return internal(
                        "a Reshape of a tensor non-zero at padding maps its padded cube as the \
                         identity",
                    );
                }
                y.copy_from_slice(input(0)?);
            }
            Op::Transpose { perm } => {
                let (x, values) = (cube_in(0)?, input(0)?);
                for (i, slot) in y.iter_mut().enumerate() {
                    let mut coords = vec![0; x.dims.len()];
                    for (k, &axis) in perm.iter().enumerate() {
                        coords[usize::from(axis)] = out.coord(i, k);
                    }
                    *slot = values[x.index(&coords)];
                }
            }
            Op::Slice { axis, start, len } => {
                let (x, values, axis) = (cube_in(0)?, input(0)?, usize::from(*axis));
                for (i, slot) in y.iter_mut().enumerate() {
                    let mut coords = out.coords(i);
                    // A selection's padded rows along its axis are 0.
                    if coords[axis] < *len as usize {
                        coords[axis] += *start as usize;
                        *slot = values[x.index(&coords)];
                    }
                }
            }
            Op::GatherPublic { index } => {
                let (x, values) = (cube_in(0)?, input(0)?);
                let rows = self.constant(*index)?;
                for (i, slot) in y.iter_mut().enumerate() {
                    let mut coords = out.coords(i);
                    if let Some(&row) = rows.get(coords[0]) {
                        coords[0] = usize::try_from(row).map_err(|_| StageError::Internal {
                            rule: "a gathered row exists",
                        })?;
                        *slot = values[x.index(&coords)];
                    }
                }
            }
            Op::Expand { axis, len } => {
                let (x, values, axis) = (cube_in(0)?, input(0)?, usize::from(*axis));
                for (i, slot) in y.iter_mut().enumerate() {
                    let mut coords = out.coords(i);
                    // 0 past its length, as its coefficient `LT(r_a, len)` says.
                    if coords.remove(axis) < *len as usize {
                        *slot = values[x.index(&coords)];
                    }
                }
            }
            Op::Sum { axis } | Op::SumSquares { axis } => {
                let (x, values, axis) = (cube_in(0)?, input(0)?, usize::from(*axis));
                let square = matches!(node.op, Op::SumSquares { .. });
                for (i, slot) in y.iter_mut().enumerate() {
                    let mut coords = out.coords(i);
                    coords.insert(axis, 0);
                    let mut total = 0i128;
                    for k in 0..1usize << x.bits[axis] {
                        coords[axis] = k;
                        let v = i128::from(values[x.index(&coords)]);
                        total += if square { v * v } else { v };
                    }
                    *slot = fit(total)?;
                }
            }
            Op::MatMulAct { transpose_rhs } => {
                let (a, b) = (input(0)?, input(1)?);
                let (ca, cb) = (cube_in(0)?, cube_in(1)?);
                let rank = ca.dims.len();
                let inner = 1usize << ca.bits[rank - 1];
                for (i, slot) in y.iter_mut().enumerate() {
                    let coords = out.coords(i);
                    let (row, column) = (coords[rank - 2], coords[rank - 1]);
                    let mut left = coords.clone();
                    let mut right = coords;
                    let mut total = 0i128;
                    for l in 0..inner {
                        (left[rank - 2], left[rank - 1]) = (row, l);
                        (right[rank - 2], right[rank - 1]) = match transpose_rhs {
                            true => (column, l),
                            false => (l, column),
                        };
                        if let Some(slots) = &graph.slots
                            && ca.dims[rank - 2] == slots.token_len() as usize
                            && cb.dims[rank - 2] == slots.token_len() as usize
                        {
                            // RFC-0071: the query and right operand share their request index.
                            right[rank - 2] +=
                                row / slots.slot_len as usize * slots.slot_len as usize;
                        }
                        total += i128::from(a[ca.index(&left)]) * i128::from(b[cb.index(&right)]);
                    }
                    *slot = fit(total)?;
                }
            }
            Op::MatMulWeight { .. } => {
                let (x, w) = (input(0)?, input(1)?);
                let (cx, cw) = (cube_in(0)?, cube_in(1)?);
                let cols = 1usize << cx.last_bits();
                for (i, slot) in y.iter_mut().enumerate() {
                    let mut coords = out.coords(i);
                    let o = coords.pop().unwrap_or(0);
                    coords.push(0);
                    let last = coords.len() - 1;
                    let mut total = 0i128;
                    for c in 0..cols {
                        coords[last] = c;
                        total +=
                            i128::from(x[cx.index(&coords)]) * i128::from(w[cw.index(&[o, c])]);
                    }
                    *slot = fit(total)?;
                }
            }
            Op::Rotate { rope } => {
                let x = input(0)?;
                let spec = self.tables.iter().find(|spec| spec.id == *rope).ok_or(
                    StageError::Internal {
                        rule: "a Rotate's table is the graph's",
                    },
                )?;
                let table = RopeTable::from_spec(spec).map_err(|_| StageError::Internal {
                    rule: "a Rotate's table regenerates",
                })?;
                let positions = witness.values(node.inputs[1]).ok_or(StageError::Internal {
                    rule: "the witness holds the positions",
                })?;
                let width = out.dims.last().copied().unwrap_or(0);
                if !width.is_power_of_two() {
                    return internal(
                        "a Rotate of a tensor non-zero at padding pairs halves of a power of two",
                    );
                }
                let half = width / 2;
                for (i, slot) in y.iter_mut().enumerate() {
                    // Its padded rows are 0, as its kernel is (RFC-0037 'Rotate').
                    let row = out.coord(i, 0);
                    let Some(&position) = positions.get(row).filter(|_| row < out.dims[0]) else {
                        continue;
                    };
                    let k = i & (width - 1);
                    let base = i - k;
                    let frequency = u32::try_from(k % half).unwrap_or(u32::MAX);
                    let position = u32::try_from(position).map_err(|_| StageError::Internal {
                        rule: "a position is a non-negative 32-bit integer",
                    })?;
                    let (c, s) =
                        table
                            .constants()
                            .at(position, frequency)
                            .ok_or(StageError::Internal {
                                rule: "a Rotate's positions and frequencies are its table's",
                            })?;
                    let (p, q) = (
                        i128::from(x[base + k % half]),
                        i128::from(x[base + k % half + half]),
                    );
                    let (c, s) = (i128::from(c), i128::from(s));
                    *slot = fit(if k < half {
                        p * c - q * s
                    } else {
                        p * s + q * c
                    })?;
                }
            }
            Op::Requant { shift, lo, hi } => {
                let x = input(0)?;
                let shifts =
                    graph
                        .shift_classes
                        .get(shift.0 as usize)
                        .ok_or(StageError::Internal {
                            rule: "a shift class set the graph holds",
                        })?;
                let classes = if shifts.len() > 1 {
                    Some(input(1)?)
                } else {
                    None
                };
                for (i, slot) in y.iter_mut().enumerate() {
                    let class = class_at(classes, i, out.last_bits(), shifts.len())?;
                    let r = u32::from(shifts[class].clamp(1, 63));
                    let shifted = (i128::from(x[i]) + (1i128 << (r - 1))) >> r;
                    *slot = fit(shifted.clamp(i128::from(*lo), i128::from(*hi)))?;
                }
            }
            Op::TableMap { table } => {
                let x = input(0)?;
                let classes = if node.inputs.len() > 1 {
                    Some(input(1)?)
                } else {
                    None
                };
                let bits = out.last_bits();
                let lookup = self.lookup(*table)?;
                let count = match &lookup.spec().kind {
                    TableKind::Silu { r_by_class, .. } | TableKind::Requant { r_by_class, .. } => {
                        r_by_class.len()
                    }
                    _ => 1,
                };
                for (i, slot) in y.iter_mut().enumerate() {
                    let class = class_at(classes, i, bits, count)?;
                    let class = u8::try_from(class).unwrap_or(u8::MAX);
                    *slot = lookup.lookup(x[i], class).ok_or(StageError::Internal {
                        rule: "a table holds every key its lookup reads at padded positions",
                    })?;
                }
            }
            Op::Bitwise { table } => {
                let (a, b) = (input(0)?, input(1)?);
                let lookup = self.lookup(*table)?;
                for (i, slot) in y.iter_mut().enumerate() {
                    // RFC-0008: the key of a `Bitwise8` lookup is `a + 2^8·b`.
                    let key = b[i]
                        .checked_mul(1 << 8)
                        .and_then(|high| high.checked_add(a[i]))
                        .ok_or(StageError::Internal {
                            rule: "a table holds every key its lookup reads at padded positions",
                        })?;
                    *slot = lookup.lookup(key, 0).ok_or(StageError::Internal {
                        rule: "a table holds every key its lookup reads at padded positions",
                    })?;
                }
            }
            Op::BitDecompose { low, high } => {
                let x = input(0)?;
                let mut outputs = Vec::with_capacity(2);
                for (k, (bits, from)) in [(*low, 0u8), (*high, *low)].into_iter().enumerate() {
                    if bits == 0 {
                        continue;
                    }
                    let t = node.outputs.get(k).ok_or(StageError::Internal {
                        rule: "a BitDecompose has an output per part",
                    })?;
                    let cube = Cube::of(&shape(graph, *t)?);
                    let mut part = zeros(cube.len())?;
                    let width = cube.last_bits();
                    for (i, slot) in part.iter_mut().enumerate() {
                        let j = i & ((1usize << width) - 1);
                        let v = x[i >> width];
                        if !(0..1i64 << (low + high)).contains(&v) {
                            return internal(
                                "a decomposed integer lies in [0, 2^{low + high}) at padded \
                                 positions",
                            );
                        }
                        // The bit slots past the part's bits are 0 (RFC-0066 'Padding').
                        if j < usize::from(bits) {
                            *slot = (v >> (usize::from(from) + j)) & 1;
                        }
                    }
                    outputs.push(part);
                }
                return Ok(outputs);
            }
            Op::BitFn { f } => {
                let ins: Vec<&[i64]> = (0..f.arity()).map(input).collect::<Result<_, _>>()?;
                // The inputs' bits at one position, secret-class: cleared when dropped (OBS-2).
                let mut at = zeroize::Zeroizing::new([0i64; 3]);
                for (i, slot) in y.iter_mut().enumerate() {
                    for (a, values) in at.iter_mut().zip(&ins) {
                        *a = values[i];
                    }
                    if at[..f.arity()].iter().any(|&b| b != 0 && b != 1) {
                        return internal(
                            "a Boolean function's inputs are bits at padded positions",
                        );
                    }
                    *slot = f.eval(&at[..f.arity()]);
                }
            }
            // Refused above when an input they read has padded positions. With none, their
            // inputs are their real values, and RFC-0035 pins their outputs to 0 at padded
            // positions: the values padded with zeros. `AssertEqual` has no output.
            Op::OneHot { .. }
            | Op::SoftmaxRow { .. }
            | Op::RsqrtRow { .. }
            | Op::Argmax
            | Op::InverseCdf { .. }
            | Op::AssertEqual => {
                return node
                    .outputs
                    .iter()
                    .map(|&t| zero_padded(graph, witness, t))
                    .collect();
            }
        }
        Ok(vec![y])
    }
}

/// The class of padded index `index`'s channel, its last axis's coordinate of `bits` bits: the
/// class vector's padded entry there, 0 at a padded channel, or 0 for one class.
fn class_at(
    classes: Option<&[i64]>,
    index: usize,
    bits: u32,
    count: usize,
) -> Result<usize, StageError> {
    let Some(classes) = classes else {
        return Ok(0);
    };
    let channel = index & ((1usize << bits) - 1);
    classes
        .get(channel)
        .and_then(|&c| usize::try_from(c).ok())
        .filter(|&c| c < count)
        .ok_or(StageError::Internal {
            rule: "a channel's class indexes its table's classes",
        })
}

fn internal<T>(rule: &'static str) -> Result<T, StageError> {
    Err(StageError::Internal { rule })
}
