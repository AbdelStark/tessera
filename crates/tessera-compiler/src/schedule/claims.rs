// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Claims and their atoms (RFC-0037 'Claims and atoms', 'Splits').

use tessera_ir::{CommitmentClass, Graph, Node, NodeId, Op, TensorId};

use super::{
    Atom, AtomClass, Claim, Coord, Factor, Kind, LayerMap, Layers, Member, NodeRef, Part,
    PoolClaim, PublicSource, PublicTerm, axis_vars, blocks, committed, public_tensors, refuse,
};
use crate::CompileError;

/// The state of a derivation: the graph, its layers, each tensor's producer and whether it is
/// public, the pool so far, and how many instances each stage's claims have given the next stage.
pub(super) struct Derivation<'a> {
    pub(super) graph: &'a Graph,
    pub(super) layers: &'a LayerMap,
    producers: Vec<Option<(NodeId, u32)>>,
    public: Vec<bool>,
    pool: Vec<PoolClaim>,
    instances: Vec<usize>,
}

impl<'a> Derivation<'a> {
    /// A derivation over `graph`, whose producers by tensor are `producers`.
    pub(super) fn new(
        graph: &'a Graph,
        layers: &'a LayerMap,
        producers: Vec<Option<(NodeId, u32)>>,
    ) -> Self {
        Self {
            graph,
            layers,
            producers,
            public: public_tensors(graph),
            pool: Vec::new(),
            instances: Vec::new(),
        }
    }

    /// Whether a claim on `kind` is on a public tensor ([`public_tensors`]): a tensor outside the
    /// layers, since no layer template holds one ([`LayerMap`]).
    pub(super) fn is_public(&self, kind: Kind) -> bool {
        match kind {
            Kind::Tensor(tensor) => self.public.get(tensor.0 as usize).copied().unwrap_or(false),
            _ => false,
        }
    }

    /// The node that produces `tensor`, `None` for a graph input: `O(1)`.
    pub(super) fn producer_of(&self, tensor: TensorId) -> Option<&'a Node> {
        let (node, _) = self.producers.get(tensor.0 as usize).copied().flatten()?;
        self.graph.nodes.get(node.0 as usize)
    }

    pub(super) fn into_pool(self) -> Vec<PoolClaim> {
        self.pool
    }

    /// The variables of `kind`, its layer variables aside.
    pub(super) fn vars(&self, kind: Kind) -> u32 {
        blocks(&self.layers.spec(self.graph, kind).shape)
            .iter()
            .sum()
    }

    /// The node that produces `kind`, as the schedule names it, when one does.
    pub(super) fn producer(&self, kind: Kind) -> Option<(NodeRef, &'a Node)> {
        match kind {
            Kind::Layer { position, .. } => {
                let node = &self.graph.nodes[self.layers.node(position, 0).0 as usize];
                Some((NodeRef::Position(position), node))
            }
            Kind::Tensor(tensor) => self
                .producer_of(tensor)
                .map(|node| (NodeRef::Node(node.id), node)),
            Kind::LayerInput | Kind::LayerParameter { .. } => None,
        }
    }

    /// The kind of input `input` of `node`, with the layer weights a claim on it takes when the
    /// node's own claim weighs the layers by `layers`.
    pub(super) fn input(
        &self,
        node: NodeRef,
        graph_node: &Node,
        input: usize,
        layers: &Layers,
    ) -> Result<(Kind, Layers), CompileError> {
        match node {
            NodeRef::Position(position) => {
                let kind = self
                    .layers
                    .input(position, u32::try_from(input).unwrap_or(u32::MAX));
                if matches!(kind, Kind::Tensor(_)) {
                    return refuse(
                        Some(graph_node.id),
                        "a claim in a layer reaches a tensor that every layer shares",
                    );
                }
                Ok((kind, layers.clone()))
            }
            NodeRef::Node(_) => Ok(self.layers.outside(graph_node.inputs[input])),
        }
    }

    /// The claim of stage `stage` on `kind` at `point` with layer weights `layers`, its `index`-th:
    /// its atoms and public terms, its selections combined ([`combine`]), and its parts, which
    /// take the next stage's instances and the pool's claims.
    pub(super) fn claim(
        &mut self,
        stage: u16,
        index: usize,
        kind: Kind,
        point: Vec<Coord>,
        layers: Layers,
    ) -> Result<Claim, CompileError> {
        let mut terms = Terms::default();
        self.substitute(kind, point.clone(), layers.clone(), Vec::new(), &mut terms)?;
        let atoms = combine(terms.atoms, |atom| self.selects(atom))?;
        let (split, parts) = self.parts(stage, index, &atoms)?;
        Ok(Claim {
            kind,
            point,
            layers,
            atoms,
            public: terms.public,
            split,
            parts,
        })
    }

    /// What a virtual atom on a selection's output selects: the input tensor's kind and the layer
    /// weights a claim on it takes, and the selected axis, the outermost for a `GatherPublic`.
    /// `None` for an atom of any other relation (RFC-0072 'Combined selections').
    fn selects(&self, atom: &Atom) -> Result<Option<Selected>, CompileError> {
        let Some((node, graph_node)) = self.producer(atom.kind) else {
            return Ok(None);
        };
        let axis = match graph_node.op {
            Op::Slice { axis, .. } => axis,
            Op::GatherPublic { .. } => 0,
            _ => return Ok(None),
        };
        let (input, layers) = self.input(node, graph_node, 0, &atom.layers)?;
        Ok(Some(Selected {
            input,
            layers,
            axis,
        }))
    }

    /// Maps a claim on `kind` at `point`, its coefficient `product` so far, onto atoms and public
    /// terms through the linear nodes (RFC-0037 'Claims and atoms', 'Head layouts'; RFC-0010
    /// 'Relations'; RFC-0072 'Public tensors').
    fn substitute(
        &self,
        kind: Kind,
        point: Vec<Coord>,
        layers: Layers,
        product: Vec<Factor>,
        terms: &mut Terms,
    ) -> Result<(), CompileError> {
        let spec = self.layers.spec(self.graph, kind);
        let class = match kind {
            Kind::LayerInput => Some(CommitmentClass::Advice),
            _ => committed(spec.kind),
        };
        if let Some(commitment) = class {
            merge(
                &mut terms.atoms,
                Atom {
                    kind,
                    point,
                    layers,
                    coefficient: vec![product],
                    class: AtomClass::Committed(commitment),
                    combined: Vec::new(),
                },
            );
            return Ok(());
        }
        // A public tensor's atom is public: the verifier computes the tensor and moves the atom
        // to the claim's value, so the walk stops at it (RFC-0072 'Public tensors' 'Their
        // claims'). No layer template holds one, and a claim in a layer on a tensor every layer
        // shares is refused before it, so it is a tensor outside the layers.
        if self
            .public
            .get(spec.id.0 as usize)
            .copied()
            .unwrap_or(false)
        {
            let (Kind::Tensor(tensor), Layers::None) = (kind, &layers) else {
                return refuse(None, "a claim on a public tensor is outside the layers");
            };
            merge_public(
                &mut terms.public,
                PublicTerm {
                    source: PublicSource::Tensor(tensor),
                    point,
                    coefficient: vec![product],
                },
            );
            return Ok(());
        }
        let Some((node, graph_node)) = self.producer(kind) else {
            return refuse(
                None,
                "a claim reaches a graph input that is neither committed nor public",
            );
        };
        let id = Some(graph_node.id);
        let input = |i: usize| self.input(node, graph_node, i, &layers);
        let shape =
            |i: usize| -> &[u32] { &self.graph.tensors[graph_node.inputs[i].0 as usize].shape };
        let output = spec.shape.as_slice();
        match &graph_node.op {
            Op::Reshape => {
                if !reshape_is_identity(shape(0), output) {
                    return refuse(
                        id,
                        "a Reshape merges or splits only axes whose lengths are powers of two, the \
                         leading axis aside",
                    );
                }
                let (kind, layers) = input(0)?;
                self.substitute(kind, point, layers, product, terms)
            }
            Op::Transpose { perm } => {
                let offsets = offsets(output);
                let vars = blocks(output);
                let rank = output.len();
                let mut mapped = Vec::with_capacity(point.len());
                for axis in (0..rank).rev() {
                    let Some(k) = perm.iter().position(|&p| usize::from(p) == axis) else {
                        return refuse(id, "a Transpose permutes its input's axes");
                    };
                    let block = vars[rank - 1 - k] as usize;
                    mapped.extend_from_slice(&point[offsets[k]..offsets[k] + block]);
                }
                let (kind, layers) = input(0)?;
                self.substitute(kind, mapped, layers, product, terms)
            }
            Op::Expand { axis, len } => {
                let offsets = offsets(output);
                let axis = usize::from(*axis);
                let block = axis_vars(*len) as usize;
                let dropped = point[offsets[axis]..offsets[axis] + block].to_vec();
                let mut kept = point;
                kept.drain(offsets[axis]..offsets[axis] + block);
                let mut product = product;
                if !len.is_power_of_two() {
                    product.push(Factor::Lt {
                        coords: dropped,
                        len: *len,
                    });
                }
                let (kind, layers) = input(0)?;
                self.substitute(kind, kept, layers, product, terms)
            }
            Op::Sum { axis } => {
                let x = shape(0);
                let axis = usize::from(*axis);
                let block = axis_vars(x[axis]);
                let at = offsets(x)[axis];
                let mut placed = point;
                placed.splice(at..at, (0..block).map(|_| Coord::Half));
                let mut product = product;
                if block > 0 {
                    product.push(Factor::Pow2(block));
                }
                let (kind, layers) = input(0)?;
                self.substitute(kind, placed, layers, product, terms)
            }
            Op::Add | Op::Sub => {
                if shape(0) != output || shape(1) != output {
                    return refuse(id, "the operands of an add have its output's shape");
                }
                let (a, a_layers) = input(0)?;
                self.substitute(a, point.clone(), a_layers, product.clone(), terms)?;
                let mut product = product;
                if graph_node.op == Op::Sub {
                    product.push(Factor::Minus);
                }
                let (b, b_layers) = input(1)?;
                self.substitute(b, point, b_layers, product, terms)
            }
            Op::ConstMul { c } if self.scalar(*c) => {
                let mut product = product;
                product.push(Factor::Constant(node));
                let (kind, layers) = input(0)?;
                self.substitute(kind, point, layers, product, terms)
            }
            // `ỹ(r) = x̃(r) + ĉ(r)`: `a·ỹ(r) = v` becomes `a·x̃(r) = v − a·ĉ(r)`, the shift a public
            // term at the claim's point (RFC-0072 'Linear operations that are not coordinate
            // maps'). No layer template holds a `ConstAdd`, so no layer weights arise.
            Op::ConstAdd { .. } => {
                let NodeRef::Node(shifted) = node else {
                    return refuse(id, "a ConstAdd is outside the layers");
                };
                let (kind, layers) = input(0)?;
                self.substitute(kind, point.clone(), layers, product.clone(), terms)?;
                merge_public(
                    &mut terms.public,
                    PublicTerm {
                        source: PublicSource::Shift(shifted),
                        point,
                        coefficient: vec![product],
                    },
                );
                Ok(())
            }
            // A sumcheck relation's output: a `ConstMul` by a constant per channel, the selections
            // `Slice` and `GatherPublic` and `BitFn` among them (RFC-0072 'Linear operations
            // that are not coordinate maps', 'The selection relation'; RFC-0066 '`BitFn`').
            Op::MatMulWeight { .. }
            | Op::MatMulAct { .. }
            | Op::Mul { .. }
            | Op::SumSquares { .. }
            | Op::Rotate { .. }
            | Op::ConstMul { .. }
            | Op::Slice { .. }
            | Op::GatherPublic { .. }
            | Op::BitFn { .. } => {
                merge(
                    &mut terms.atoms,
                    Atom {
                        kind,
                        point,
                        layers,
                        coefficient: vec![product],
                        class: AtomClass::Virtual,
                        combined: Vec::new(),
                    },
                );
                Ok(())
            }
            Op::OneHot { .. }
            | Op::Requant { .. }
            | Op::TableMap { .. }
            | Op::SoftmaxRow { .. }
            | Op::RsqrtRow { .. }
            | Op::Argmax
            | Op::InverseCdf { .. }
            | Op::Bitwise { .. }
            | Op::BitDecompose { .. }
            | Op::AssertEqual => refuse(id, "a gadget's outputs are committed"),
        }
    }

    /// Whether constant `c` is one scalar.
    fn scalar(&self, c: tessera_ir::ConstRef) -> bool {
        self.graph
            .constants
            .get(c.0 as usize)
            .is_some_and(|v| v.len() == 1)
    }

    /// The parts of the claim `index` of stage `stage` with `atoms` (RFC-0037 'Splits'), and
    /// whether the stage splits it.
    ///
    /// The claim's public terms take no part: the verifier moves them to its value first. A claim
    /// is then ready when it is one virtual atom of coefficient 1, or committed atoms of one
    /// commitment at one point. Otherwise each virtual atom is a part, and each group of committed
    /// atoms of one commitment at one point, in the place of its first atom. A claim with no atom,
    /// public terms alone, has no part and is not split.
    fn parts(
        &mut self,
        stage: u16,
        index: usize,
        atoms: &[Atom],
    ) -> Result<(bool, Vec<Part>), CompileError> {
        if atoms.is_empty() {
            return Ok((false, Vec::new()));
        }
        for atom in atoms {
            self.check_uniform(atom)?;
        }
        if let [atom] = atoms
            && atom.class == AtomClass::Virtual
            && atom.is_unit()
        {
            let instance = self.next_instance(stage);
            return Ok((false, vec![Part::Instance { atom: 0, instance }]));
        }
        // The committed groups, by commitment and point, in the order of their first atoms.
        let mut groups: Vec<(CommitmentClass, &[Coord], Vec<usize>)> = Vec::new();
        for (i, atom) in atoms.iter().enumerate() {
            if let AtomClass::Committed(commitment) = atom.class {
                match groups
                    .iter_mut()
                    .find(|(c, p, _)| *c == commitment && *p == atom.point.as_slice())
                {
                    Some((_, _, members)) => members.push(i),
                    None => groups.push((commitment, &atom.point, vec![i])),
                }
            }
        }
        let virtuals = atoms.iter().any(|a| a.class == AtomClass::Virtual);
        if !virtuals && let [(commitment, point, members)] = groups.as_slice() {
            let claim = self.pool_claim(stage, index, *commitment, point, members, atoms);
            let part = Part::Pool {
                atoms: members.clone(),
                claim,
            };
            return Ok((false, vec![part]));
        }
        let mut parts = Vec::new();
        for (i, atom) in atoms.iter().enumerate() {
            match atom.class {
                AtomClass::Virtual => {
                    let instance = self.next_instance(stage);
                    parts.push(Part::Instance { atom: i, instance });
                }
                AtomClass::Committed(_) => {
                    if let Some((commitment, point, members)) =
                        groups.iter().find(|(_, _, members)| members[0] == i)
                    {
                        let claim =
                            self.pool_claim(stage, index, *commitment, point, members, atoms);
                        parts.push(Part::Pool {
                            atoms: members.clone(),
                            claim,
                        });
                    }
                }
            }
        }
        Ok((true, parts))
    }

    /// Checks that a virtual atom stacked at a layer point is weighed by `eq` alone: a constant of a
    /// template position that differs between layers would weigh its layers otherwise. A combined
    /// selection's members' coefficients count as its own.
    fn check_uniform(&self, atom: &Atom) -> Result<(), CompileError> {
        if atom.class != AtomClass::Virtual || !matches!(atom.layers, Layers::Eq(_)) {
            return Ok(());
        }
        let members = atom.combined.iter().flat_map(|m| &m.coefficient);
        for factor in atom.coefficient.iter().chain(members).flatten() {
            if let Factor::Constant(NodeRef::Position(position)) = factor
                && !self.layers.uniform(*position)
            {
                return refuse(
                    Some(self.layers.node(*position, 0)),
                    "a relation's output stacked over the layers is scaled by one constant in \
                     every layer",
                );
            }
        }
        Ok(())
    }

    /// The next instance of the stage after `stage`.
    fn next_instance(&mut self, stage: u16) -> usize {
        let stage = usize::from(stage);
        if self.instances.len() <= stage {
            self.instances.resize(stage + 1, 0);
        }
        let next = self.instances[stage];
        self.instances[stage] += 1;
        next
    }

    /// Adds the pool claim of `members` of `atoms`, part of claim `index` of stage `stage`. A
    /// leaf stage's per-layer atoms arrive stacked at the layer point `ρ_1`.
    fn pool_claim(
        &mut self,
        stage: u16,
        index: usize,
        commitment: CommitmentClass,
        point: &[Coord],
        members: &[usize],
        atoms: &[Atom],
    ) -> usize {
        let rho = || (0..self.layers.vars).map(Coord::Layer).collect::<Vec<_>>();
        let atoms = members
            .iter()
            .map(|&i| {
                let mut atom = atoms[i].clone();
                if atom.layers == Layers::Each {
                    atom.layers = Layers::Eq(rho());
                }
                atom
            })
            .collect();
        self.pool.push(PoolClaim {
            commitment,
            point: point.to_vec(),
            atoms,
            stage,
            claim: index,
        });
        self.pool.len() - 1
    }
}

/// A claim's atoms and public terms, as substitution reaches them.
#[derive(Default)]
struct Terms {
    atoms: Vec<Atom>,
    public: Vec<PublicTerm>,
}

/// What a selection's output selects: rows of `input`, a claim on which takes the layer weights
/// `layers`, along its axis `axis`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Selected {
    input: Kind,
    layers: Layers,
    axis: u8,
}

/// RFC-0072 'Combined selections' on one claim's `atoms`, `selects` saying what a virtual atom
/// selects, `None` for one that is no selection's output. Two or more virtual atoms that select
/// rows of one input tensor along one axis, at one point with one set of layer weights, become
/// one atom at the first one's place: of coefficient 1, the first one's kind, point and layer
/// weights, and the members ([`Atom::combined`]) in the order substitution reached them, each
/// with its coefficient. A single selection atom is not combined, and every other atom keeps its
/// place. Atoms of one kind at one point have merged already, so the members are distinct
/// tensors. The rule reads one claim's atoms alone: no claim merges with another ('No merging
/// across claims').
fn combine(
    atoms: Vec<Atom>,
    selects: impl Fn(&Atom) -> Result<Option<Selected>, CompileError>,
) -> Result<Vec<Atom>, CompileError> {
    let mut keys = Vec::with_capacity(atoms.len());
    for atom in &atoms {
        keys.push(match atom.class {
            AtomClass::Virtual => selects(atom)?,
            AtomClass::Committed(_) => None,
        });
    }
    let same = |i: usize, j: usize| {
        keys[j].is_some()
            && keys[j] == keys[i]
            && atoms[j].point == atoms[i].point
            && atoms[j].layers == atoms[i].layers
    };
    let mut taken = vec![false; atoms.len()];
    let mut combined = Vec::with_capacity(atoms.len());
    for i in 0..atoms.len() {
        if taken[i] {
            continue;
        }
        let members: Vec<usize> = (i..atoms.len())
            .filter(|&j| !taken[j] && same(i, j))
            .collect();
        if members.len() < 2 {
            combined.push(atoms[i].clone());
            continue;
        }
        for &j in &members {
            taken[j] = true;
        }
        let first = &atoms[i];
        combined.push(Atom {
            kind: first.kind,
            point: first.point.clone(),
            layers: first.layers.clone(),
            coefficient: vec![Vec::new()],
            class: AtomClass::Virtual,
            combined: members
                .iter()
                .map(|&j| Member {
                    kind: atoms[j].kind,
                    coefficient: atoms[j].coefficient.clone(),
                })
                .collect(),
        });
    }
    Ok(combined)
}

/// Adds `term` to `public`, merged into a term of its source at its point, in the first one's
/// place: the verifier evaluates each source once per point (RFC-0072 'Public tensors' 'Once per
/// point').
fn merge_public(public: &mut Vec<PublicTerm>, term: PublicTerm) {
    match public
        .iter_mut()
        .find(|t| t.source == term.source && t.point == term.point)
    {
        Some(existing) => existing.coefficient.extend(term.coefficient),
        None => public.push(term),
    }
}

/// Adds `atom` to `atoms`, merged into an atom on its kind at its point with its layer weights.
fn merge(atoms: &mut Vec<Atom>, atom: Atom) {
    match atoms
        .iter_mut()
        .find(|a| a.kind == atom.kind && a.point == atom.point && a.layers == atom.layers)
    {
        Some(existing) => existing.coefficient.extend(atom.coefficient),
        None => atoms.push(atom),
    }
}

/// The offset of each axis's block in a point of `shape`, by axis: the last axis's block first.
pub(super) fn offsets(shape: &[u32]) -> Vec<usize> {
    let mut offsets = vec![0; shape.len()];
    let mut at = 0;
    for (axis, &len) in shape.iter().enumerate().rev() {
        offsets[axis] = at;
        at += axis_vars(len) as usize;
    }
    offsets
}

/// Whether a `Reshape` of `x` into `y` maps the coordinates of their per-dimension padded
/// extensions as the identity (RFC-0037 'Head layouts'): taking both shapes from the last axis in
/// groups of equal products, a group of one axis on each side keeps its axis, and any other group
/// holds only axes whose lengths are powers of two, its leading axis aside.
fn reshape_is_identity(x: &[u32], y: &[u32]) -> bool {
    let (mut i, mut j) = (x.len(), y.len());
    loop {
        let ones = |shape: &[u32]| shape.iter().all(|&len| len == 1);
        if i == 0 || j == 0 {
            return ones(&x[..i]) && ones(&y[..j]);
        }
        let (end_x, end_y) = (i, j);
        i -= 1;
        j -= 1;
        let (mut px, mut py) = (u64::from(x[i]), u64::from(y[j]));
        while px != py {
            if px < py {
                let Some(next) = i.checked_sub(1) else {
                    return false;
                };
                i = next;
                px *= u64::from(x[i]);
            } else {
                let Some(next) = j.checked_sub(1) else {
                    return false;
                };
                j = next;
                py *= u64::from(y[j]);
            }
        }
        let (group_x, group_y) = (&x[i..end_x], &y[j..end_y]);
        if group_x.len() == 1 && group_y.len() == 1 {
            continue;
        }
        let fits = |group: &[u32]| group.iter().skip(1).all(|len| len.is_power_of_two());
        if !fits(group_x) || !fits(group_y) {
            return false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Atom, AtomClass, Coord, Factor, Kind, Layers, Selected, combine, reshape_is_identity,
    };
    use tessera_ir::{CommitmentClass, TensorId};

    /// RFC-0072 'Combined selections' on atoms built here: the rule reads the selected tensor and
    /// axis, the point and the layer weights, and nothing else; committed atoms and virtual atoms
    /// of other relations keep their places, and the combination takes its first member's.
    #[test]
    fn selections_combine_on_one_tensor_axis_point_and_layer_weights() {
        let x = Kind::Tensor(TensorId(0));
        let (p, q) = (vec![Coord::Leaf { site: 0, index: 0 }], vec![Coord::Half]);
        let atom = |t: u32, point: &[Coord], layers: Layers, committed: bool| Atom {
            kind: Kind::Tensor(TensorId(t)),
            point: point.to_vec(),
            layers,
            coefficient: vec![vec![Factor::Pow2(t)]],
            class: if committed {
                AtomClass::Committed(CommitmentClass::Advice)
            } else {
                AtomClass::Virtual
            },
            combined: Vec::new(),
        };
        let v = |t: u32, point: &[Coord]| atom(t, point, Layers::None, false);
        // Tensors 1, 2, 3 and 6 select `x`'s axis 0, 4 its axis 1, 5 another tensor's axis 0; 7
        // is no selection's output.
        let selects = |a: &Atom| {
            let Kind::Tensor(TensorId(t)) = a.kind else {
                return Ok(None);
            };
            let on = |input: Kind, axis: u8| {
                Some(Selected {
                    input,
                    layers: Layers::None,
                    axis,
                })
            };
            Ok(match t {
                1 | 2 | 3 | 6 => on(x, 0),
                4 => on(x, 1),
                5 => on(Kind::Tensor(TensorId(9)), 0),
                _ => None,
            })
        };
        let kinds = |atoms: &[Atom]| -> Vec<(Kind, usize)> {
            atoms.iter().map(|a| (a.kind, a.combined.len())).collect()
        };
        let t = |t: u32| Kind::Tensor(TensorId(t));
        let run = |atoms: Vec<Atom>| combine(atoms, selects).expect("combined");
        // One tensor, one axis, one point, one set of layer weights: one atom of coefficient 1,
        // its members' coefficients theirs.
        let both = run(vec![v(1, &p), v(2, &p)]);
        assert_eq!(kinds(&both), [(t(1), 2)]);
        assert!(both[0].is_unit());
        assert_eq!(both[0].combined[1].coefficient, [vec![Factor::Pow2(2)]]);
        // In the first member's place, the other atoms in theirs.
        let committed = atom(3, &p, Layers::None, true);
        let placed = run(vec![v(7, &p), v(1, &p), committed, v(2, &p)]);
        assert_eq!(kinds(&placed), [(t(7), 0), (t(1), 2), (t(3), 0)]);
        // Two combinations at two points.
        let two = run(vec![v(1, &p), v(3, &q), v(2, &p), v(6, &q)]);
        assert_eq!(kinds(&two), [(t(1), 2), (t(3), 2)]);
        // None of these combine.
        let apart = [
            (
                "different layer weights",
                vec![v(1, &p), atom(2, &p, Layers::Eq(Vec::new()), false)],
            ),
            ("different axes", vec![v(1, &p), v(4, &p)]),
            ("different tensors", vec![v(1, &p), v(5, &p)]),
            ("different points", vec![v(1, &p), v(2, &q)]),
            ("a single atom", vec![v(1, &p)]),
            (
                "committed atoms",
                vec![
                    atom(1, &p, Layers::None, true),
                    atom(2, &p, Layers::None, true),
                ],
            ),
            ("no selection", vec![v(7, &p), v(1, &p)]),
        ];
        for (name, atoms) in apart {
            assert_eq!(run(atoms.clone()), atoms, "{name}");
        }
    }

    /// The lowering's reshapes, and ones that move a padded axis.
    #[test]
    fn reshapes_are_identities_exactly_on_power_of_two_groups() {
        let identities: [(&[u32], &[u32]); 11] = [
            (&[5, 256], &[5, 2, 128]),
            (&[5, 2, 128], &[5, 256]),
            (&[3, 96], &[3, 6, 16]),
            (&[3, 6, 16], &[3, 96]),
            (&[3, 48], &[3, 3, 16]),
            (&[3, 3, 16], &[3, 48]),
            (&[5, 1, 128], &[5, 128]),
            (&[1, 2, 5, 128], &[2, 5, 128]),
            (&[3, 2, 5, 128], &[6, 5, 128]),
            (&[6], &[3, 2]),
            (&[4, 128, 1], &[4, 128]),
        ];
        for (x, y) in identities {
            assert!(reshape_is_identity(x, y), "{x:?} → {y:?}");
        }
        let moves: [(&[u32], &[u32]); 6] = [
            (&[5, 3], &[15]),
            (&[6], &[2, 3]),
            (&[2, 3, 128], &[6, 128]),
            (&[5, 96], &[5, 2, 48]),
            (&[3, 12], &[3, 2, 6]),
            (&[3, 288], &[3, 6, 48]),
        ];
        for (x, y) in moves {
            assert!(!reshape_is_identity(x, y), "{x:?} → {y:?}");
        }
    }
}
