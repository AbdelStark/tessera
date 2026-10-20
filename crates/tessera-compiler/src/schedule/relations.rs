// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reduction stages: each instance's relation, its variables, and the claims its evaluations
//! leave (RFC-0010 'Relations', RFC-0037 'Stages and their instances', 'Rotate').

use tessera_ir::{Broadcast, Node, NodeKind, Op, OpClass};

use super::claims::Derivation;
use super::{
    Claim, Coord, Instance, Kind, Layers, NodeRef, Part, Relation, Source, Stage, axis_vars,
    blocks, refuse,
};
use crate::CompileError;

/// An instance before its stage's rounds are known.
struct Planned<'a> {
    source: Source,
    relation: Relation,
    node: NodeRef,
    graph_node: &'a Node,
    kind: Kind,
    point: Vec<Coord>,
    layers: Layers,
    vars: u32,
}

/// Stage `index`: an instance per virtual part of the stage before's claims `previous`, in order,
/// then in stage 1 the equality zero-checks in node order, and the claims their evaluations leave.
/// It has no instances when nothing remains to prove.
pub(super) fn stage(
    derivation: &mut Derivation<'_>,
    index: u16,
    previous: &[Claim],
) -> Result<Stage, CompileError> {
    let lambda = derivation.layers.vars;
    let mut planned = Vec::new();
    for (c, claim) in previous.iter().enumerate() {
        for (p, part) in claim.parts.iter().enumerate() {
            let Part::Instance { atom, .. } = part else {
                continue;
            };
            let atom = &claim.atoms[*atom];
            // The leaf stage's per-layer claims are stacked at the layer point ρ_1.
            let layers = match &atom.layers {
                Layers::Each => Layers::Eq((0..lambda).map(Coord::Layer).collect()),
                other => other.clone(),
            };
            let Some((node, graph_node)) = derivation.producer(atom.kind) else {
                return refuse(None, "a virtual atom is a node's output");
            };
            let relation = relation(graph_node)?;
            let vars = instance_vars(derivation, graph_node, relation, &layers)?;
            planned.push(Planned {
                source: Source::Part { claim: c, part: p },
                relation,
                node,
                graph_node,
                kind: atom.kind,
                point: atom.point.clone(),
                layers,
                vars,
            });
        }
    }
    let first_zero = planned.len();
    if index == 1 {
        for (node, graph_node, kind) in zero_checks(derivation) {
            let layers = match node {
                NodeRef::Position(_) => Layers::Eq(Vec::new()),
                NodeRef::Node(_) => Layers::None,
            };
            let relation = if matches!(graph_node.op, Op::OneHot { .. }) {
                Relation::Booleanity
            } else {
                Relation::Equality
            };
            let vars = instance_vars(derivation, graph_node, relation, &layers)?;
            planned.push(Planned {
                source: Source::ZeroCheck,
                relation,
                node,
                graph_node,
                kind,
                point: Vec::new(),
                layers,
                vars,
            });
        }
    }
    let rounds = planned.iter().map(|p| p.vars).max().unwrap_or(0);
    let zero_vars = planned[first_zero..]
        .iter()
        .map(|p| p.vars)
        .max()
        .unwrap_or(0);
    let mut instances = Vec::with_capacity(planned.len());
    let mut claims = Vec::new();
    for mut plan in planned {
        let challenges: Vec<Coord> = (rounds - plan.vars..rounds)
            .map(|round| Coord::Round {
                stage: index,
                round,
            })
            .collect();
        if plan.source == Source::ZeroCheck {
            // Its coordinates of the zero-check point: the last n_i, the layer's the highest.
            let zero: Vec<Coord> = (zero_vars - plan.vars..zero_vars)
                .map(Coord::Zero)
                .collect();
            let own = zero.len() - stacked(&plan.layers, lambda);
            if let Layers::Eq(rho) = &mut plan.layers {
                *rho = zero[own..].to_vec();
            }
            plan.point = zero[..own].to_vec();
        }
        let start = claims.len();
        for (kind, point, layers) in evaluations(derivation, &plan, &challenges)? {
            let claim = derivation.claim(index, claims.len(), kind, point, layers)?;
            claims.push(claim);
        }
        instances.push(Instance {
            source: plan.source,
            relation: plan.relation,
            node: plan.node,
            kind: plan.kind,
            point: plan.point,
            layers: plan.layers,
            vars: plan.vars,
            challenges,
            claims: start..claims.len(),
        });
    }
    Ok(Stage {
        index,
        instances,
        rounds,
        zero_vars,
        claims,
    })
}

/// The layer variables of an instance whose claim weighs the layers by `layers`.
fn stacked(layers: &Layers, lambda: u32) -> usize {
    match layers {
        Layers::Eq(_) => lambda as usize,
        _ => 0,
    }
}

/// The relation that proves `node`'s virtual output.
fn relation(node: &Node) -> Result<Relation, CompileError> {
    Ok(match node.op {
        Op::MatMulWeight { .. } => Relation::MatMulWeight,
        Op::MatMulAct { transpose_rhs } => Relation::MatMulAct {
            transposed: transpose_rhs,
        },
        Op::Mul { broadcast } => Relation::Mul(broadcast),
        Op::SumSquares { .. } => Relation::SumSquares,
        Op::Rotate { .. } => Relation::Rotate,
        Op::ConstMul { .. } => Relation::ScaleChannels,
        _ => {
            return refuse(
                Some(node.id),
                "a virtual atom is a sumcheck relation's output",
            );
        }
    })
}

/// The nodes whose committed output stage 1 proves by an equality zero-check, in node order: each
/// linear or degree-2 node other than a matmul with a committed output ('The committed first
/// checkpoint'), a template position once.
fn zero_checks<'a>(derivation: &Derivation<'a>) -> Vec<(NodeRef, &'a Node, Kind)> {
    let layers = derivation.layers;
    let mut checks = Vec::new();
    for node in &derivation.graph.nodes {
        let reference = match layers.position(node.id) {
            Some((0, position)) => NodeRef::Position(position),
            Some(_) => continue,
            None => NodeRef::Node(node.id),
        };
        let checked = matches!(node.op, Op::OneHot { .. })
            || (node.op.class() != OpClass::Gadget
                && !matches!(node.op, Op::MatMulWeight { .. } | Op::MatMulAct { .. }));
        for (output, &t) in (0u32..).zip(&node.outputs) {
            let committed = matches!(
                derivation.graph.tensors[t.0 as usize].kind,
                NodeKind::Committed { .. }
            );
            if checked && committed {
                let kind = match reference {
                    NodeRef::Position(position) => Kind::Layer { position, output },
                    NodeRef::Node(_) => Kind::Tensor(t),
                };
                checks.push((reference, node, kind));
            }
        }
    }
    checks
}

/// The variables of an instance of `relation` on `node` whose claim weighs the layers by
/// `layers`: the relation's summed and free variables, then the layer variables.
fn instance_vars(
    derivation: &Derivation<'_>,
    node: &Node,
    relation: Relation,
    layers: &Layers,
) -> Result<u32, CompileError> {
    if matches!(layers, Layers::One(_) | Layers::Each) {
        return refuse(
            Some(node.id),
            "a relation proves a claim on every layer or on none",
        );
    }
    let lambda = u32::try_from(stacked(layers, derivation.layers.vars)).unwrap_or(u32::MAX);
    let shape = |t: tessera_ir::TensorId| derivation.graph.tensors[t.0 as usize].shape.as_slice();
    let vars = |t| blocks(shape(t)).iter().sum::<u32>();
    let own = match relation {
        Relation::MatMulWeight => match (shape(node.inputs[0]), shape(node.inputs[1])) {
            ([_, inner], [_, inner_w]) if inner == inner_w => axis_vars(*inner),
            _ => {
                return refuse(
                    Some(node.id),
                    "a MatMulWeight multiplies [n, in] by [out, in]",
                );
            }
        },
        Relation::MatMulAct { .. } => match shape(node.inputs[0]) {
            [heads, _, inner] => axis_vars(*inner) + axis_vars(*heads),
            _ => {
                return refuse(
                    Some(node.id),
                    "a MatMulAct multiplies [heads, a, j] matrices",
                );
            }
        },
        Relation::SumSquares => {
            let x = shape(node.inputs[0]);
            if !matches!(node.op, Op::SumSquares { axis } if usize::from(axis) + 1 == x.len()) {
                return refuse(Some(node.id), "a SumSquares runs over the last axis");
            }
            vars(node.inputs[0])
        }
        Relation::Rotate => {
            let x = shape(node.inputs[0]);
            if !x
                .last()
                .is_some_and(|&len| len >= 2 && len.is_power_of_two())
            {
                return refuse(
                    Some(node.id),
                    "a Rotate pairs a last axis whose length is a power of two",
                );
            }
            vars(node.inputs[0])
        }
        Relation::Mul(_) | Relation::ScaleChannels | Relation::Equality | Relation::Booleanity => {
            vars(node.outputs[0])
        }
    };
    Ok(own + lambda)
}

/// The claims an instance's evaluations leave, in its relation's order: the kind, the point and
/// the layer weights of each (RFC-0010 'Relations', RFC-0037 'Rotate').
fn evaluations(
    derivation: &Derivation<'_>,
    plan: &Planned<'_>,
    challenges: &[Coord],
) -> Result<Vec<(Kind, Vec<Coord>, Layers)>, CompileError> {
    let node = plan.graph_node;
    let id = Some(node.id);
    let lambda = stacked(&plan.layers, derivation.layers.vars);
    let (own, rho) = challenges.split_at(challenges.len() - lambda);
    let layers = match plan.layers {
        Layers::Eq(_) => Layers::Eq(rho.to_vec()),
        _ => Layers::None,
    };
    let shape = |i: usize| {
        derivation.graph.tensors[node.inputs[i].0 as usize]
            .shape
            .as_slice()
    };
    // The kind of input `i` and the layer weights of a claim on it.
    let input = |i: usize| derivation.input(plan.node, node, i, &layers);
    let at = |i: usize, point: Vec<Coord>| -> Result<(Kind, Vec<Coord>, Layers), CompileError> {
        let (kind, layers) = input(i)?;
        Ok((kind, point, layers))
    };
    let target = plan.point.as_slice();
    Ok(match plan.relation {
        Relation::MatMulWeight => {
            // Y [n, out] at r_o ‖ r_t; the instance's point r'_i ‖ ρ'.
            let out = axis_vars(shape(1)[0]) as usize;
            let (r_o, r_t) = target.split_at(out);
            vec![at(1, [own, r_o].concat())?, at(0, [own, r_t].concat())?]
        }
        Relation::MatMulAct { transposed } => {
            // Z [h, a, c] at r_c ‖ r_a ‖ r_h; the instance's point r'_j ‖ r'_h ‖ ρ'.
            let [_, rows, inner] = shape(0) else {
                return refuse(id, "a MatMulAct multiplies [heads, a, j] matrices");
            };
            let (r_j, r_h) = own.split_at(axis_vars(*inner) as usize);
            let columns = target.len() - axis_vars(*rows) as usize - r_h.len();
            let (r_c, rest) = target.split_at(columns);
            let r_a = &rest[..axis_vars(*rows) as usize];
            let right = if transposed {
                [r_j, r_c, r_h].concat()
            } else {
                [r_c, r_j, r_h].concat()
            };
            vec![at(0, [r_j, r_a, r_h].concat())?, at(1, right)?]
        }
        Relation::Mul(broadcast) => vec![
            at(0, own.to_vec())?,
            at(1, broadcast_point(shape(0), shape(1), broadcast, own, id)?)?,
        ],
        Relation::SumSquares | Relation::Rotate | Relation::ScaleChannels => {
            vec![at(0, own.to_vec())?]
        }
        Relation::Booleanity => vec![(plan.kind, own.to_vec(), layers.clone())],
        Relation::Equality => {
            let output = (plan.kind, own.to_vec(), layers.clone());
            let mut claims = vec![output];
            match &node.op {
                Op::ConstMul { .. } => claims.push(at(0, own.to_vec())?),
                Op::Add | Op::Sub => {
                    claims.push(at(0, own.to_vec())?);
                    claims.push(at(1, own.to_vec())?);
                }
                Op::Mul { broadcast } => {
                    claims.push(at(0, own.to_vec())?);
                    claims.push(at(
                        1,
                        broadcast_point(shape(0), shape(1), *broadcast, own, id)?,
                    )?);
                }
                _ => {
                    return refuse(
                        id,
                        "a committed output's equality zero-check is of a ConstMul, Add, Sub or Mul",
                    );
                }
            }
            claims
        }
    })
}

/// The point of `Mul`'s broadcast operand `b` within the product's point `own` over `a`'s shape.
fn broadcast_point(
    a: &[u32],
    b: &[u32],
    broadcast: Broadcast,
    own: &[Coord],
    id: Option<tessera_ir::NodeId>,
) -> Result<Vec<Coord>, CompileError> {
    match broadcast {
        // `b` is `a`'s leading axes: the highest coordinates.
        Broadcast::Trailing if a.starts_with(b) => {
            let below: u32 = blocks(&a[b.len()..]).iter().sum();
            Ok(own[below as usize..].to_vec())
        }
        // `b` is `a`'s last axis: the lowest coordinates.
        Broadcast::Leading if a.last().is_some_and(|last| b == [*last]) => {
            Ok(own[..axis_vars(b[0]) as usize].to_vec())
        }
        _ => refuse(
            id,
            "a Mul's second operand is its first's leading axes or its last axis",
        ),
    }
}
