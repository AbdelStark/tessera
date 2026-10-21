// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Each instance's relation over its stage's tables (RFC-0010 'Relations', RFC-0037 'Rotate'),
//! and the claims its evaluations leave.

use tessera_codec::SecretBuf;
use tessera_compiler::schedule::{Instance as Planned, Layers, NodeRef, Relation, Stage};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_ir::{Broadcast, Node, Op};
use tessera_sumcheck::{
    Claim, Elementwise, ElementwiseShape, ElementwiseTables, Input, Instance, MatMulAct,
    MatMulActShape, MatMulActTables, MatMulWeight, MatMulWeightShape, MatMulWeightTables,
    RightLayout, Rotate, RotateTables, Term,
};
use tessera_tables::RopeTable;

use super::tables::{axis_vars, zeros};
use super::{StageError, StageProver, WitnessSource};

/// A stage's tables: the witness's, secret, and public ones such as RoPE's constants.
#[derive(Default)]
pub(super) struct Arena {
    pub(super) secret: Vec<SecretBuf<Goldilocks>>,
    pub(super) public: Vec<Vec<Goldilocks>>,
}

impl Arena {
    fn secret(&mut self, table: SecretBuf<Goldilocks>) -> usize {
        self.secret.push(table);
        self.secret.len() - 1
    }

    fn public(&mut self, table: Vec<Goldilocks>) -> usize {
        self.public.push(table);
        self.public.len() - 1
    }
}

/// How an elementwise relation reads the arena.
pub(super) enum Read {
    Operand(usize),
    /// A broadcast operand laid out over the rows: an operand to the relation, whose claim is
    /// the broadcast's at its own coordinates.
    Spread(usize),
    Broadcast {
        table: usize,
        low_vars: usize,
    },
    Public(usize),
}

/// An instance's relation, its tables named in the arena.
pub(super) enum Plan {
    MatMulWeight {
        w: usize,
        x: usize,
        shape: MatMulWeightShape,
    },
    MatMulAct {
        a: usize,
        b: usize,
        layout: RightLayout,
        shape: MatMulActShape,
    },
    Elementwise {
        inputs: Vec<Read>,
        terms: Vec<Term<GoldilocksExt2>>,
        shape: ElementwiseShape,
        /// Drop the vocabulary coordinates of the common zero point for row checks.
        point_from: usize,
    },
    Rotate {
        x: usize,
        cos: usize,
        sin: usize,
        bit: usize,
        vars: usize,
    },
}

impl Plan {
    /// The evaluations whose claims are on an operand laid out over the rows: in the relation's
    /// claim order, which leaves the public inputs out.
    pub(super) fn spread(&self) -> Vec<usize> {
        let Self::Elementwise { inputs, .. } = self else {
            return Vec::new();
        };
        inputs
            .iter()
            .filter(|read| !matches!(read, Read::Public(_)))
            .enumerate()
            .filter(|(_, read)| matches!(read, Read::Spread(_)))
            .map(|(claim, _)| claim)
            .collect()
    }
}

/// An instance's tables, built at its claim's point.
pub(super) enum Built<'a> {
    MatMulWeight(MatMulWeightTables<'a, GoldilocksExt2>),
    MatMulAct(MatMulActTables<'a, GoldilocksExt2>),
    Elementwise(ElementwiseTables<'a, GoldilocksExt2>),
    Rotate(RotateTables<'a, GoldilocksExt2>),
}

impl Built<'_> {
    /// The instance `prove_batch` proves.
    pub(super) fn instance(&self) -> Instance<'_, GoldilocksExt2> {
        match self {
            Self::MatMulWeight(t) => t.instance(),
            Self::MatMulAct(t) => t.instance(),
            Self::Elementwise(t) => t.instance(),
            Self::Rotate(t) => t.instance(),
        }
    }

    /// The claims its evaluations leave, in its relation's order.
    pub(super) fn claims(
        &self,
        point: &[GoldilocksExt2],
        evals: &[GoldilocksExt2],
    ) -> Result<Vec<Claim<GoldilocksExt2>>, StageError> {
        Ok(match self {
            Self::MatMulWeight(t) => {
                let claims = t.claims(point, evals)?;
                vec![claims.w, claims.x]
            }
            Self::MatMulAct(t) => {
                let claims = t.claims(point, evals)?;
                vec![claims.a, claims.b]
            }
            Self::Elementwise(t) => t.claims(point, evals)?,
            Self::Rotate(t) => vec![t.claim(point, evals)?],
        })
    }
}

/// Builds every plan's tables at its point: `points[i]` is plan `i`'s claim point, its layer
/// coordinates included. Returns the instances' tables and their degrees.
pub(super) fn build<'a>(
    arena: &'a Arena,
    plans: &[Plan],
    points: &[Vec<GoldilocksExt2>],
) -> Result<(Vec<Built<'a>>, usize), StageError> {
    let mut built = Vec::with_capacity(plans.len());
    let mut degree = 1;
    for (plan, point) in plans.iter().zip(points) {
        let tables = match plan {
            Plan::MatMulWeight { w, x, shape } => {
                let relation = MatMulWeight::new(&arena.secret[*w], &arena.secret[*x], *shape)?;
                degree = degree.max(relation.degree());
                Built::MatMulWeight(relation.tables(point)?)
            }
            Plan::MatMulAct {
                a,
                b,
                layout,
                shape,
            } => {
                let relation =
                    MatMulAct::new(&arena.secret[*a], &arena.secret[*b], *layout, *shape)?;
                degree = degree.max(relation.degree());
                Built::MatMulAct(relation.tables(point)?)
            }
            Plan::Elementwise {
                inputs,
                terms,
                shape,
                point_from,
            } => {
                let point = point.get(*point_from..).ok_or(StageError::Internal {
                    rule: "a row zero-check projects its common point",
                })?;
                let inputs = inputs
                    .iter()
                    .map(|read| match *read {
                        Read::Operand(t) | Read::Spread(t) => Input::Operand(&arena.secret[t][..]),
                        Read::Broadcast { table, low_vars } => Input::Broadcast {
                            operand: &arena.secret[table][..],
                            low_vars,
                        },
                        Read::Public(t) => Input::Public(&arena.public[t][..]),
                    })
                    .collect();
                let relation = Elementwise::new(*shape, inputs, terms.clone())?;
                degree = degree.max(relation.degree());
                Built::Elementwise(relation.tables(point)?)
            }
            Plan::Rotate {
                x,
                cos,
                sin,
                bit,
                vars,
            } => {
                let relation = Rotate::new(
                    &arena.secret[*x],
                    &arena.public[*cos],
                    &arena.public[*sin],
                    *bit,
                    *vars,
                )?;
                degree = degree.max(relation.degree());
                Built::Rotate(relation.tables(point)?)
            }
        };
        built.push(tables);
    }
    Ok((built, degree))
}

impl<W: WitnessSource + ?Sized> StageProver<'_, W> {
    /// The plan of `instance` of `stage`, its tables added to `arena`; the node it proves.
    pub(super) fn plan(
        &self,
        stage: &Stage,
        instance: &Planned,
        arena: &mut Arena,
    ) -> Result<Plan, StageError> {
        let layers = &self.schedule.layers;
        let node = self.node(instance.node);
        let stacked = matches!(instance.layers, Layers::Eq(_));
        let lambda = if stacked { layers.vars as usize } else { 0 };
        let claims = &stage.claims[instance.claims.clone()];
        let operand = |i: usize| -> Result<SecretBuf<Goldilocks>, StageError> {
            let claim = claims.get(i).ok_or(StageError::Internal {
                rule: "an instance's evaluations name its operands",
            })?;
            self.operand(claim.kind, &claim.layers)
        };
        let shape = |t: usize| self.lowering.graph.tensors[t].shape.as_slice();
        let input_shape = |i: usize| shape(node.inputs[i].0 as usize);
        let output = shape(node.outputs[0].0 as usize);
        let vars = |s: &[u32]| Self::vars_of(s) as usize;
        Ok(match instance.relation {
            Relation::Native { .. } => {
                return internal("native relations are built by their composed backend");
            }
            Relation::MatMulWeight => {
                let (&[n, inner], &[out, _]) = (input_shape(0), input_shape(1)) else {
                    return internal("a MatMulWeight multiplies [n, in] by [out, in]");
                };
                let shape = MatMulWeightShape {
                    layer_vars: lambda,
                    token_vars: axis_vars(n) as usize,
                    out_vars: axis_vars(out) as usize,
                    in_vars: axis_vars(inner) as usize,
                };
                Plan::MatMulWeight {
                    w: arena.secret(operand(0)?),
                    x: arena.secret(operand(1)?),
                    shape,
                }
            }
            Relation::MatMulAct { transposed } => {
                let (&[heads, rows, inner], &[_, _, columns]) = (input_shape(0), output) else {
                    return internal("a MatMulAct multiplies [heads, a, j] matrices");
                };
                let shape = MatMulActShape {
                    stack_vars: axis_vars(heads) as usize + lambda,
                    row_vars: axis_vars(rows) as usize,
                    inner_vars: axis_vars(inner) as usize,
                    col_vars: axis_vars(columns) as usize,
                };
                let layout = if transposed {
                    RightLayout::Transposed
                } else {
                    RightLayout::Direct
                };
                Plan::MatMulAct {
                    a: arena.secret(operand(0)?),
                    b: arena.secret(operand(1)?),
                    layout,
                    shape,
                }
            }
            Relation::Mul(broadcast) => {
                let a = arena.secret(operand(0)?);
                let b = self.broadcast(
                    arena,
                    operand(1)?,
                    input_shape(0),
                    input_shape(1),
                    broadcast,
                    stacked,
                )?;
                Plan::Elementwise {
                    point_from: 0,
                    inputs: vec![Read::Operand(a), b],
                    terms: vec![Term::new(GoldilocksExt2::ONE, vec![0, 1])],
                    shape: ElementwiseShape::elementwise(vars(output) + lambda),
                }
            }
            Relation::SumSquares => {
                let x = input_shape(0);
                let last = x.last().map_or(0, |&len| axis_vars(len) as usize);
                Plan::Elementwise {
                    point_from: 0,
                    inputs: vec![Read::Operand(arena.secret(operand(0)?))],
                    terms: vec![Term::new(GoldilocksExt2::ONE, vec![0, 0])],
                    shape: ElementwiseShape {
                        vars: vars(x) + lambda,
                        summed_vars: last,
                    },
                }
            }
            Relation::Rotate => {
                let x = input_shape(0);
                let (cos, sin) = self.rope(node, x, stacked)?;
                let bit = x.last().map_or(0, |&len| axis_vars(len) as usize) - 1;
                Plan::Rotate {
                    x: arena.secret(operand(0)?),
                    cos: arena.public(cos),
                    sin: arena.public(sin),
                    bit,
                    vars: vars(x) + lambda,
                }
            }
            Relation::ScaleChannels => {
                return internal("the stages do not prove a ConstMul by one constant per channel");
            }
            Relation::Booleanity => {
                let h = Read::Operand(arena.secret(operand(0)?));
                Plan::Elementwise {
                    point_from: 0,
                    inputs: vec![h],
                    terms: vec![
                        Term::new(GoldilocksExt2::ONE, vec![0]),
                        Term::new(-GoldilocksExt2::ONE, vec![0, 0]),
                    ],
                    shape: ElementwiseShape::elementwise(instance.vars as usize),
                }
            }
            Relation::OneHotVocabulary | Relation::OneHotCount | Relation::OneHotIndex => {
                let &[positions, vocab] = output else {
                    return internal("OneHot has token and vocabulary axes");
                };
                let width = vocab.next_power_of_two() as usize;
                let vocab_vars = axis_vars(vocab) as usize;
                let length = 1usize << instance.vars;
                let one = GoldilocksExt2::ONE;
                let h = Read::Operand(arena.secret(operand(0)?));
                let inverse_width =
                    GoldilocksExt2::from(Goldilocks::from_u64_reduced(width as u64))
                        .inverse()
                        .ok_or(StageError::Internal {
                            rule: "the padded vocabulary is nonzero in the field",
                        })?;
                match instance.relation {
                    Relation::OneHotVocabulary => {
                        let outside = (0..length)
                            .map(|i| {
                                if i % width >= vocab as usize {
                                    Goldilocks::ONE
                                } else {
                                    Goldilocks::ZERO
                                }
                            })
                            .collect();
                        Plan::Elementwise {
                            inputs: vec![h, Read::Public(arena.public(outside))],
                            terms: vec![Term::new(one, vec![0, 1])],
                            shape: ElementwiseShape::elementwise(instance.vars as usize),
                            point_from: 0,
                        }
                    }
                    Relation::OneHotCount => {
                        let selector = (0..length)
                            .map(|i| {
                                if i / width < positions as usize {
                                    Goldilocks::ONE
                                } else {
                                    Goldilocks::ZERO
                                }
                            })
                            .collect();
                        Plan::Elementwise {
                            inputs: vec![h, Read::Public(arena.public(selector))],
                            terms: vec![
                                Term::new(one, vec![0]),
                                Term::new(-inverse_width, vec![1]),
                            ],
                            shape: ElementwiseShape {
                                vars: instance.vars as usize,
                                summed_vars: vocab_vars,
                            },
                            point_from: vocab_vars,
                        }
                    }
                    Relation::OneHotIndex => {
                        let token = Read::Broadcast {
                            table: arena.secret(operand(1)?),
                            low_vars: vocab_vars,
                        };
                        let index = (0..length)
                            .map(|i| Goldilocks::from_u64_reduced((i % width) as u64))
                            .collect();
                        Plan::Elementwise {
                            inputs: vec![h, token, Read::Public(arena.public(index))],
                            terms: vec![
                                Term::new(one, vec![0, 2]),
                                Term::new(-inverse_width, vec![1]),
                            ],
                            shape: ElementwiseShape {
                                vars: instance.vars as usize,
                                summed_vars: vocab_vars,
                            },
                            point_from: vocab_vars,
                        }
                    }
                    _ => return internal("the selected OneHot relation has one canonical plan"),
                }
            }
            Relation::Equality => self.equality(arena, instance, node, &operand, stacked)?,
        })
    }

    /// The graph node of `node`: layer 0's at a template position.
    pub(super) fn node(&self, node: NodeRef) -> &Node {
        let id = match node {
            NodeRef::Node(id) => id,
            NodeRef::Position(position) => self.schedule.layers.node(position, 0),
        };
        &self.lowering.graph.nodes[id.0 as usize]
    }

    /// `Mul`'s second operand as its relation reads it: a broadcast window of the product's
    /// variables, or, for one channel per layer, laid out over the rows.
    fn broadcast(
        &self,
        arena: &mut Arena,
        b: SecretBuf<Goldilocks>,
        a_shape: &[u32],
        b_shape: &[u32],
        broadcast: Broadcast,
        stacked: bool,
    ) -> Result<Read, StageError> {
        let a_vars = Self::vars_of(a_shape) as usize;
        let b_vars = Self::vars_of(b_shape) as usize;
        Ok(match broadcast {
            // `b` is `a`'s leading axes: the highest variables, the layer's above them.
            Broadcast::Trailing => Read::Broadcast {
                table: arena.secret(b),
                low_vars: a_vars - b_vars,
            },
            Broadcast::Leading if !stacked => Read::Broadcast {
                table: arena.secret(b),
                low_vars: 0,
            },
            // The channels are the lowest variables and the layer's the highest: no one window.
            Broadcast::Leading => {
                let layers = 1usize << self.schedule.layers.vars;
                let (block, rows) = (1usize << b_vars, 1usize << (a_vars - b_vars));
                let mut full = zeros(layers * rows * block)?;
                for layer in 0..layers {
                    let channels = &b[layer * block..(layer + 1) * block];
                    for row in 0..rows {
                        let at = (layer * rows + row) * block;
                        full[at..at + block].copy_from_slice(channels);
                    }
                }
                Read::Spread(arena.secret(full))
            }
        })
    }

    /// RoPE's constants `C` and `S` for a `Rotate` of `x` (RFC-0037 'Rotate'): entry `k` belongs
    /// to the pair whose index, bit `b` removed, is `k`, `b` the last axis's top bit. They depend
    /// on the row's position and the frequency only: the same for every head and every layer, and
    /// 0 on a padded row.
    fn rope(
        &self,
        node: &Node,
        x: &[u32],
        stacked: bool,
    ) -> Result<(Vec<Goldilocks>, Vec<Goldilocks>), StageError> {
        let Op::Rotate { rope } = node.op else {
            return internal("a Rotate names its RoPE table");
        };
        let spec = self
            .lowering
            .tables
            .iter()
            .find(|spec| spec.id == rope)
            .ok_or(StageError::Internal {
                rule: "a Rotate's table is the lowering's",
            })?;
        let table = RopeTable::from_spec(spec).map_err(|_| StageError::Internal {
            rule: "a Rotate's table regenerates",
        })?;
        let positions = self
            .witness
            .values(node.inputs[1])
            .ok_or(StageError::Internal {
                rule: "the witness holds the positions",
            })?;
        let [rows, heads, dim] = *x else {
            return internal("a Rotate turns [n, heads, d_head]");
        };
        let (bit, head_vars, row_vars) = (
            axis_vars(dim) as usize - 1,
            axis_vars(heads) as usize,
            axis_vars(rows) as usize,
        );
        let layer_vars = if stacked {
            self.schedule.layers.vars as usize
        } else {
            0
        };
        let len = 1usize << (bit + head_vars + row_vars + layer_vars);
        let (mut cos, mut sin) = (vec![Goldilocks::ZERO; len], vec![Goldilocks::ZERO; len]);
        for k in 0..len {
            let frequency = k & ((1 << bit) - 1);
            let row = (k >> (bit + head_vars)) & ((1 << row_vars) - 1);
            if row >= rows as usize {
                continue;
            }
            let position = u32::try_from(positions[row]).map_err(|_| StageError::Internal {
                rule: "a position is a non-negative 32-bit integer",
            })?;
            let frequency = u32::try_from(frequency).unwrap_or(u32::MAX);
            let (c, s) = table
                .constants()
                .at(position, frequency)
                .ok_or(StageError::Internal {
                    rule: "a Rotate's positions and frequencies are its table's",
                })?;
            cos[k] = Goldilocks::from_i64(c);
            sin[k] = Goldilocks::from_i64(s);
        }
        Ok((cos, sin))
    }

    /// The equality zero-check `0 = Σ_x eq(z, x)·(Ỹ(x) − f(x))` of `node`'s committed output
    /// (RFC-0037 'The committed first checkpoint'), its evaluations `Y`'s then `f`'s inputs'.
    fn equality(
        &self,
        arena: &mut Arena,
        instance: &Planned,
        node: &Node,
        operand: &dyn Fn(usize) -> Result<SecretBuf<Goldilocks>, StageError>,
        stacked: bool,
    ) -> Result<Plan, StageError> {
        let layers = &self.schedule.layers;
        let lambda = if stacked { layers.vars as usize } else { 0 };
        let output = &self.lowering.graph.tensors[node.outputs[0].0 as usize].shape;
        let shape = ElementwiseShape::elementwise(Self::vars_of(output) as usize + lambda);
        let one = GoldilocksExt2::ONE;
        let y = Read::Operand(arena.secret(operand(0)?));
        let (inputs, terms) = match node.op {
            Op::ConstMul { .. } => {
                let x = Read::Operand(arena.secret(operand(1)?));
                let constant = |layer: u32| -> Result<i64, StageError> {
                    let id = match instance.node {
                        NodeRef::Node(id) => id,
                        NodeRef::Position(position) => layers.node(position, layer),
                    };
                    match self.lowering.graph.nodes[id.0 as usize].op {
                        Op::ConstMul { c } => self
                            .lowering
                            .graph
                            .constants
                            .get(c.0 as usize)
                            .and_then(|v| match v[..] {
                                [scalar] => Some(scalar),
                                _ => None,
                            })
                            .ok_or(StageError::Internal {
                                rule: "a committed ConstMul's constant is one scalar",
                            }),
                        _ => internal("a template position has one operation"),
                    }
                };
                let uniform = match instance.node {
                    NodeRef::Node(_) => true,
                    NodeRef::Position(position) => layers.uniform(position),
                };
                if uniform {
                    let c = GoldilocksExt2::from(Goldilocks::from_i64(constant(0)?));
                    (
                        vec![y, x],
                        vec![Term::new(one, vec![0]), Term::new(-c, vec![1])],
                    )
                } else {
                    // Each layer's own constant: a public table constant within each layer.
                    let block = 1usize << Self::vars_of(output);
                    let mut table = vec![Goldilocks::ZERO; block << layers.vars];
                    for layer in 0..layers.count {
                        let value = Goldilocks::from_i64(constant(layer)?);
                        let at = layer as usize * block;
                        table[at..at + block].fill(value);
                    }
                    let c = Read::Public(arena.public(table));
                    (
                        vec![y, x, c],
                        vec![Term::new(one, vec![0]), Term::new(-one, vec![2, 1])],
                    )
                }
            }
            Op::Add | Op::Sub => {
                let a = Read::Operand(arena.secret(operand(1)?));
                let b = Read::Operand(arena.secret(operand(2)?));
                let sign = if node.op == Op::Sub { one } else { -one };
                (
                    vec![y, a, b],
                    vec![
                        Term::new(one, vec![0]),
                        Term::new(-one, vec![1]),
                        Term::new(sign, vec![2]),
                    ],
                )
            }
            Op::Mul { broadcast } => {
                let a_shape = &self.lowering.graph.tensors[node.inputs[0].0 as usize].shape;
                let b_shape = &self.lowering.graph.tensors[node.inputs[1].0 as usize].shape;
                let a = Read::Operand(arena.secret(operand(1)?));
                let b = self.broadcast(arena, operand(2)?, a_shape, b_shape, broadcast, stacked)?;
                (
                    vec![y, a, b],
                    vec![Term::new(one, vec![0]), Term::new(-one, vec![1, 2])],
                )
            }
            _ => {
                return internal(
                    "a committed output's zero-check is of a ConstMul, Add, Sub or Mul",
                );
            }
        };
        Ok(Plan::Elementwise {
            point_from: 0,
            inputs,
            terms,
            shape,
        })
    }
}

fn internal<T>(rule: &'static str) -> Result<T, StageError> {
    Err(StageError::Internal { rule })
}
