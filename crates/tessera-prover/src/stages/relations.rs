// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Each instance's relation over its stage's tables (RFC-0010 'Relations', RFC-0037 'Rotate',
//! RFC-0072 'The bit zero-checks'), and the claims its evaluations leave.

use tessera_compiler::schedule::{Instance as Planned, Kind, Layers, NodeRef, Relation, Stage};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_ir::{BoolFn, Broadcast, Node, Op};
use tessera_sumcheck::{
    Claim, Elementwise, ElementwiseShape, ElementwiseTables, Input, Instance, MatMulAct,
    MatMulActShape, MatMulActTables, MatMulWeight, MatMulWeightShape, MatMulWeightTables,
    Recomposition, RecompositionTables, RightLayout, Rotate, RotateTables, Selected, Selection,
    SelectionTables, Term,
};
use tessera_tables::RopeTable;

use super::columns::Column;
use super::public::{channel_table, const_add, constant_and_shape, public_table, public_zeros};
use super::source_memory::{add, array};
use super::tables::{axis_vars, pad, zeros};
use super::{StageError, StageProver, WitnessSource};

/// A stage's tables: the witness's, secret, and public ones such as RoPE's constants.
#[derive(Default)]
pub(super) struct Arena {
    pub(super) secret: Vec<Column>,
    pub(super) public: Vec<Column>,
    pub(super) streaming: bool,
    pub(super) geometry_only: bool,
}

impl Arena {
    fn secret(&mut self, table: impl Into<Column>) -> usize {
        self.secret.push(table.into());
        self.secret.len() - 1
    }

    fn public(&mut self, table: impl Into<Column>) -> usize {
        self.public.push(table.into());
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

/// An input of a relation's node: public, read as `Public` from its table over its own variables,
/// or the table of the instance's claim on it (RFC-0072 'Public tensors' 'As an operand').
enum Operand {
    Public(Column),
    Claimed(Column),
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
    /// A selection (RFC-0072 'The selection relation'): `x` along an axis of `axis` variables
    /// with `low` below and `high` above, the output's axis of `rows`, and the kernel's members,
    /// each its coefficient and rows: one of coefficient 1 for a single selection.
    Selection {
        x: usize,
        low: usize,
        axis: usize,
        high: usize,
        rows: usize,
        members: Vec<(GoldilocksExt2, Vec<usize>)>,
    },
    /// A `BitDecompose`'s recomposition (RFC-0072 'The bit zero-checks'): `x`, an operand or
    /// public, its `b` and `h`, and its `eq` at the instance's point from `point_from`, the bit
    /// variables' count `k`, on.
    Recomposition {
        x: Read,
        b: usize,
        h: Option<usize>,
        low: u8,
        high: u8,
        rows: usize,
        point_from: usize,
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
    Selection(SelectionTables<GoldilocksExt2>),
    Recomposition(RecompositionTables<GoldilocksExt2>),
}

impl Built<'_> {
    /// The instance `prove_batch` proves.
    pub(super) fn instance(&self) -> Instance<'_, GoldilocksExt2> {
        match self {
            Self::MatMulWeight(t) => t.instance(),
            Self::MatMulAct(t) => t.instance(),
            Self::Elementwise(t) => t.instance(),
            Self::Rotate(t) => t.instance(),
            Self::Selection(t) => t.instance(),
            Self::Recomposition(t) => t.instance(),
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
            Self::Selection(t) => vec![t.claim(point, evals)?],
            Self::Recomposition(t) => t.claims(point, evals)?,
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
                let relation = MatMulWeight::new(
                    arena.secret[*w].resident()?,
                    arena.secret[*x].resident()?,
                    *shape,
                )?;
                degree = degree.max(relation.degree());
                Built::MatMulWeight(relation.tables(point)?)
            }
            Plan::MatMulAct {
                a,
                b,
                layout,
                shape,
            } => {
                let relation = MatMulAct::new(
                    arena.secret[*a].resident()?,
                    arena.secret[*b].resident()?,
                    *layout,
                    *shape,
                )?;
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
                    .map(|read| {
                        Ok(match *read {
                            Read::Operand(t) | Read::Spread(t) => {
                                Input::Operand(arena.secret[t].resident()?)
                            }
                            Read::Broadcast { table, low_vars } => Input::Broadcast {
                                operand: arena.secret[table].resident()?,
                                low_vars,
                            },
                            Read::Public(t) => Input::Public(arena.public[t].resident()?),
                        })
                    })
                    .collect::<Result<Vec<_>, StageError>>()?;
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
                    arena.secret[*x].resident()?,
                    arena.public[*cos].resident()?,
                    arena.public[*sin].resident()?,
                    *bit,
                    *vars,
                )?;
                degree = degree.max(relation.degree());
                Built::Rotate(relation.tables(point)?)
            }
            // The claim's point on the output, `r_lo ‖ r_a ‖ r_hi`, builds the kernel and
            // contracts `x` at its other axes' coordinates.
            Plan::Selection {
                x,
                low,
                axis,
                high,
                rows,
                members,
            } => {
                let relation = Selection::new(arena.secret[*x].resident()?, *low, *axis, *high)?;
                degree = degree.max(relation.degree());
                let kernel: Vec<Selected<'_, GoldilocksExt2>> = members
                    .iter()
                    .map(|(coefficient, index)| Selected {
                        coefficient: *coefficient,
                        index,
                    })
                    .collect();
                Built::Selection(relation.tables(point, *rows, &kernel)?)
            }
            Plan::Recomposition {
                x,
                b,
                h,
                low,
                high,
                rows,
                point_from,
            } => {
                let x = match *x {
                    Read::Operand(t) => Input::Operand(arena.secret[t].resident()?),
                    Read::Public(t) => Input::Public(arena.public[t].resident()?),
                    Read::Spread(_) | Read::Broadcast { .. } => {
                        return internal("a recomposition reads its integer whole");
                    }
                };
                let relation = Recomposition::new(
                    x,
                    arena.secret[*b].resident()?,
                    h.map(|t| arena.secret[t].resident()).transpose()?,
                    *low,
                    *high,
                    *rows,
                )?;
                degree = degree.max(relation.degree());
                let point = point.get(*point_from..).ok_or(StageError::Internal {
                    rule: "a recomposition's eq takes its point's highest coordinates",
                })?;
                Built::Recomposition(relation.tables(point)?)
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
        let streaming = arena.streaming;
        let operand = |i: usize| -> Result<Column, StageError> {
            let claim = claims.get(i).ok_or(StageError::Internal {
                rule: "an instance's evaluations name its operands",
            })?;
            self.column(claim.kind, &claim.layers, streaming)
        };
        let shape = |t: usize| self.graph.tensors[t].shape.as_slice();
        let input_shape = |i: usize| shape(node.inputs[i].0 as usize);
        // `AssertEqual` has no output.
        let output = node
            .outputs
            .first()
            .map_or(&[][..], |t| shape(t.0 as usize));
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
                let slots = self.graph.slots.as_ref().filter(|slots| {
                    rows == slots.token_len() && input_shape(1).get(1) == Some(&slots.token_len())
                });
                let shape = MatMulActShape {
                    stack_vars: axis_vars(heads) as usize
                        + lambda
                        + slots.map_or(0, |s| s.request_bits() as usize),
                    row_vars: slots.map_or(axis_vars(rows), |s| s.slot_bits()) as usize,
                    inner_vars: if !transposed {
                        slots.map_or(axis_vars(inner), |s| s.slot_bits())
                    } else {
                        axis_vars(inner)
                    } as usize,
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
            // A public operand is read as `Public`, a broadcast one laid out over the product's
            // variables, and leaves no claim (RFC-0072 'Public tensors' 'As an operand').
            Relation::Mul(broadcast) => {
                let mut next = 0;
                let a = match self.input(instance, node, 0, &operand, &mut next, arena.streaming)? {
                    Operand::Public(table) => Read::Public(arena.public(table)),
                    Operand::Claimed(table) => Read::Operand(arena.secret(table)),
                };
                let b = self.mul_operand(
                    arena,
                    self.input(instance, node, 1, &operand, &mut next, arena.streaming)?,
                    input_shape(0),
                    input_shape(1),
                    broadcast,
                    stacked,
                )?;
                Plan::Elementwise {
                    point_from: 0,
                    inputs: vec![a, b],
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
                let (cos, sin) = self.rope(node, x, stacked, arena.streaming)?;
                let bit = x.last().map_or(0, |&len| axis_vars(len) as usize) - 1;
                Plan::Rotate {
                    x: arena.secret(operand(0)?),
                    cos: arena.public(cos),
                    sin: arena.public(sin),
                    bit,
                    vars: vars(x) + lambda,
                }
            }
            // RFC-0010's `Elementwise` `Ỹ(r) = Σ_x eq(r, x)·c̃(x_c)·X̃(x)`, degree 3, the constant a
            // public operand broadcast over the other axes, 0 at padded channels; one claim on
            // `X` (RFC-0072 'ConstMul by a constant per channel'). No layer template holds one.
            Relation::ScaleChannels => {
                if stacked {
                    return internal("a ConstMul by a constant per channel is outside the layers");
                }
                let x = Read::Operand(arena.secret(operand(0)?));
                let c = Read::Public(arena.public(self.channels(node, arena.streaming)?));
                Plan::Elementwise {
                    point_from: 0,
                    inputs: vec![x, c],
                    terms: vec![Term::new(GoldilocksExt2::ONE, vec![0, 1])],
                    shape: ElementwiseShape::elementwise(vars(output)),
                }
            }
            Relation::Selection => {
                if stacked {
                    return internal("a selection is outside the layers");
                }
                self.selection(stage, instance, node, arena, &operand)?
            }
            // RFC-0010's `Elementwise` with RFC-0066's polynomial, each public input read as
            // `Public` (RFC-0072 'Public tensors' 'As an operand').
            Relation::BitFn(f) => {
                let mut next = 0;
                let mut inputs = Vec::with_capacity(node.inputs.len());
                for i in 0..node.inputs.len() {
                    inputs.push(
                        match self.input(instance, node, i, &operand, &mut next, arena.streaming)? {
                            Operand::Public(table) => Read::Public(arena.public(table)),
                            Operand::Claimed(table) => Read::Operand(arena.secret(table)),
                        },
                    );
                }
                if inputs.len() != f.arity() {
                    return internal("a BitFn has its function's inputs");
                }
                Plan::Elementwise {
                    point_from: 0,
                    inputs,
                    terms: bit_terms(f),
                    shape: ElementwiseShape::elementwise(vars(output) + lambda),
                }
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
                        let outside = Column::generated(arena.streaming, length, 0, move |i| {
                            Ok(if i % width >= vocab as usize {
                                Goldilocks::ONE
                            } else {
                                Goldilocks::ZERO
                            })
                        })?;
                        Plan::Elementwise {
                            inputs: vec![h, Read::Public(arena.public(outside))],
                            terms: vec![Term::new(one, vec![0, 1])],
                            shape: ElementwiseShape::elementwise(instance.vars as usize),
                            point_from: 0,
                        }
                    }
                    Relation::OneHotCount => {
                        let slots = self.graph.slots.clone();
                        let owned = slots
                            .as_ref()
                            .map_or(Ok(0), |s| array::<u32>(s.lengths.capacity()))?;
                        let selector =
                            Column::generated(arena.streaming, length, owned, move |i| {
                                Ok(
                                    if slots
                                        .as_ref()
                                        .map_or(i / width < positions as usize, |slots| {
                                            slots.is_real_row(i / width)
                                        })
                                    {
                                        Goldilocks::ONE
                                    } else {
                                        Goldilocks::ZERO
                                    },
                                )
                            })?;
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
                        let index = Column::generated(arena.streaming, length, 0, move |i| {
                            Ok(Goldilocks::from_u64_reduced((i % width) as u64))
                        })?;
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
            Relation::InputPadding
            | Relation::BitPadding
            | Relation::Recomposition
            | Relation::AssertEqual => {
                if stacked {
                    return internal("a bit zero-check is outside the layers");
                }
                self.bits(arena, instance, node, claims, &operand)?
            }
        })
    }

    fn column(&self, kind: Kind, layers: &Layers, streaming: bool) -> Result<Column, StageError> {
        if !streaming {
            return Ok(self.operand(kind, layers)?.into());
        }
        let shape = self.shape(kind)?.to_vec();
        let block = 1usize
            .checked_shl(Self::vars_of(&shape))
            .ok_or(StageError::Memory)?;
        let (tensors, len) = match layers {
            Layers::Eq(_) => (
                (0..self.schedule.layers.count)
                    .map(|l| self.tensor(kind, l))
                    .collect::<Result<Vec<_>, _>>()?,
                block
                    .checked_shl(self.schedule.layers.vars)
                    .ok_or(StageError::Memory)?,
            ),
            Layers::One(layer) => (vec![self.tensor(kind, *layer)?], block),
            Layers::None => (vec![self.tensor(kind, 0)?], block),
            Layers::Each => return internal("an instance's operand has a fixed layer placement"),
        };
        Ok(Column::Tensor {
            tensors,
            shape,
            block,
            len,
            padded: true,
        })
    }

    /// RFC-0066's zero-checks of a `BitDecompose` or an `AssertEqual` (RFC-0072 'The bit
    /// zero-checks'), each at its coordinates of the zero-check point:
    ///
    /// - **Input padding**: `ũ·(1 − ψ̃)`, `ψ` the committed input's selector;
    /// - **Bit padding**: `b̃·(1 − LT(x_bit, n))`, `n` the output's bits;
    /// - **Recomposition**: [`Recomposition`], `x` read as `Public` when public;
    /// - **`AssertEqual`**: `ψ̃·ã − ψ̃·b̃`, each side read as `Public` when public, `ψ` the
    ///   tensors' selector.
    ///
    /// Booleanity is `OneHot`'s relation, on the output its claim names.
    fn bits(
        &self,
        arena: &mut Arena,
        instance: &Planned,
        node: &Node,
        claims: &[tessera_compiler::schedule::Claim],
        operand: &dyn Fn(usize) -> Result<Column, StageError>,
    ) -> Result<Plan, StageError> {
        let one = GoldilocksExt2::ONE;
        let shape = ElementwiseShape::elementwise(instance.vars as usize);
        // The tensor a padding check reads, its one claim's; `AssertEqual` of two public tensors
        // has none.
        let claimed = || {
            claims.first().ok_or(StageError::Internal {
                rule: "a padding check claims its tensor",
            })
        };
        Ok(match instance.relation {
            Relation::InputPadding => {
                let u = self.shape(claimed()?.kind)?.to_vec();
                Plan::Elementwise {
                    point_from: 0,
                    inputs: vec![
                        Read::Operand(arena.secret(operand(0)?)),
                        Read::Public(arena.public(Column::selector(&u, true, arena.streaming)?)),
                    ],
                    terms: vec![Term::new(one, vec![0, 1])],
                    shape,
                }
            }
            Relation::BitPadding => {
                let Op::BitDecompose { low, high } = node.op else {
                    return internal("a bit-padding check is a BitDecompose's");
                };
                let kind = claimed()?.kind;
                let tensor = self.tensor(kind, 0)?;
                let bits = if node.outputs.get(1) == Some(&tensor) {
                    high
                } else {
                    low
                };
                let output = self.shape(kind)?.to_vec();
                Plan::Elementwise {
                    point_from: 0,
                    inputs: vec![
                        Read::Operand(arena.secret(operand(0)?)),
                        Read::Public(arena.public(Column::bit_slots(
                            &output,
                            bits,
                            arena.streaming,
                        )?)),
                    ],
                    terms: vec![Term::new(one, vec![0, 1])],
                    shape,
                }
            }
            Relation::Recomposition => {
                let Op::BitDecompose { low, high } = node.op else {
                    return internal("a recomposition is a BitDecompose's");
                };
                let mut next = 0;
                let x = match self.input(instance, node, 0, operand, &mut next, arena.streaming)? {
                    Operand::Public(table) => Read::Public(arena.public(table)),
                    Operand::Claimed(table) => Read::Operand(arena.secret(table)),
                };
                let b = arena.secret(operand(next)?);
                let h = match high {
                    0 => None,
                    _ => Some(arena.secret(operand(next + 1)?)),
                };
                let rows = Self::vars_of(&self.graph.tensors[node.inputs[0].0 as usize].shape);
                let k = axis_vars(u32::from(low)).max(axis_vars(u32::from(high)));
                Plan::Recomposition {
                    x,
                    b,
                    h,
                    low,
                    high,
                    rows: rows as usize,
                    point_from: k as usize,
                }
            }
            Relation::AssertEqual => {
                let mut next = 0;
                let mut read = |arena: &mut Arena, i: usize| -> Result<Read, StageError> {
                    Ok(
                        match self.input(instance, node, i, operand, &mut next, arena.streaming)? {
                            Operand::Public(table) => Read::Public(arena.public(table)),
                            Operand::Claimed(table) => Read::Operand(arena.secret(table)),
                        },
                    )
                };
                let a = read(arena, 0)?;
                let b = read(arena, 1)?;
                let tensors = &self.graph.tensors[node.inputs[0].0 as usize].shape;
                let psi =
                    Read::Public(arena.public(Column::selector(tensors, false, arena.streaming)?));
                Plan::Elementwise {
                    point_from: 0,
                    inputs: vec![a, b, psi],
                    terms: vec![Term::new(one, vec![2, 0]), Term::new(-one, vec![2, 1])],
                    shape,
                }
            }
            _ => return internal("a bit zero-check is RFC-0066's"),
        })
    }

    /// RFC-0072's selection relation of `instance`'s node, a `Slice` or a `GatherPublic` of `x`:
    /// `x`'s table, its variables below, along and above the selected axis, the output axis's,
    /// and the kernel's members. A combined selection's ([`Atom::combined`]) are its members',
    /// each with its coefficient, all selecting along this node's axis of this node's input; a
    /// single selection's kernel is its own `S`, of coefficient 1, its atom's coefficient
    /// weighing its split value instead.
    ///
    /// [`Atom::combined`]: tessera_compiler::schedule::Atom::combined
    fn selection(
        &self,
        stage: &Stage,
        instance: &Planned,
        node: &Node,
        arena: &mut Arena,
        operand: &dyn Fn(usize) -> Result<Column, StageError>,
    ) -> Result<Plan, StageError> {
        let (axis, _) = selected(self.graph, node)?;
        let x = &self.graph.tensors[node.inputs[0].0 as usize].shape;
        let y = &self.graph.tensors[node.outputs[0].0 as usize].shape;
        let vars = |axes: &[u32]| Self::vars_of(axes) as usize;
        let atom =
            self.schedule
                .atom_of(stage.index, instance.source)
                .ok_or(StageError::Internal {
                    rule: "a selection proves a virtual part",
                })?;
        let mut members = Vec::new();
        if atom.combined.is_empty() {
            members.push((GoldilocksExt2::ONE, selected(self.graph, node)?.1));
        }
        for member in &atom.combined {
            let Kind::Tensor(tensor) = member.kind else {
                return internal("a combined selection is outside the layers");
            };
            let producer = self.producer(tensor).ok_or(StageError::Internal {
                rule: "a combined selection's member is a node's output",
            })?;
            let (member_axis, index) = selected(self.graph, producer)?;
            if member_axis != axis || producer.inputs[0] != node.inputs[0] {
                return internal("a combined selection's members select one axis of one tensor");
            }
            members.push((
                if arena.geometry_only {
                    GoldilocksExt2::ZERO
                } else {
                    self.coefficient(&member.coefficient, 0)?
                },
                index,
            ));
        }
        Ok(Plan::Selection {
            x: arena.secret(operand(0)?),
            low: vars(&x[axis + 1..]),
            axis: axis_vars(x[axis]) as usize,
            high: vars(&x[..axis]),
            rows: axis_vars(y[axis]) as usize,
            members,
        })
    }

    /// The graph node of `node`: layer 0's at a template position.
    pub(super) fn node(&self, node: NodeRef) -> &Node {
        let id = match node {
            NodeRef::Node(id) => id,
            NodeRef::Position(position) => self.schedule.layers.node(position, 0),
        };
        &self.graph.nodes[id.0 as usize]
    }

    /// Input `i` of `instance`'s node `node` as its relation reads it: a public input's table, the
    /// tensor padded per dimension (RFC-0072 'Public tensors' 'As an operand'); any other's, the
    /// table of the instance's claim `*next`, which then advances. A template position's inputs
    /// are never public: the schedule refuses a claim in a layer on a tensor every layer shares.
    fn input(
        &self,
        instance: &Planned,
        node: &Node,
        i: usize,
        operand: &dyn Fn(usize) -> Result<Column, StageError>,
        next: &mut usize,
        streaming: bool,
    ) -> Result<Operand, StageError> {
        let tensor = *node.inputs.get(i).ok_or(StageError::Internal {
            rule: "a relation's node has its inputs",
        })?;
        if matches!(instance.node, NodeRef::Node(_)) && self.public.contains(tensor) {
            if streaming {
                let mut column = self.column(Kind::Tensor(tensor), &Layers::None, true)?;
                if let Column::Tensor { padded, .. } = &mut column {
                    *padded = false;
                }
                return Ok(Operand::Public(column));
            }
            let table = public_table(self.graph, &self.public, self.witness, tensor)?;
            return Ok(Operand::Public(table.into()));
        }
        let table = operand(*next)?;
        *next += 1;
        Ok(Operand::Claimed(table))
    }

    /// `Mul`'s second operand as its relation reads it: a claimed one as [`Self::broadcast`]
    /// lays it out, a public one laid out over every variable of the product, the first operand's,
    /// since `Elementwise` has no public broadcast: `Trailing` on the highest, `Leading` on the
    /// lowest.
    fn mul_operand(
        &self,
        arena: &mut Arena,
        b: Operand,
        a_shape: &[u32],
        b_shape: &[u32],
        broadcast: Broadcast,
        stacked: bool,
    ) -> Result<Read, StageError> {
        let table = match b {
            Operand::Claimed(table) => {
                return self.broadcast(arena, table, a_shape, b_shape, broadcast, stacked);
            }
            Operand::Public(table) => table,
        };
        let (a_vars, b_vars) = (Self::vars_of(a_shape), Self::vars_of(b_shape));
        let low = match broadcast {
            Broadcast::Trailing => a_vars.checked_sub(b_vars).ok_or(StageError::Internal {
                rule: "a Mul's second operand is its first's leading axes",
            })?,
            Broadcast::Leading => 0,
        };
        if arena.streaming {
            return Ok(Read::Public(arena.public(Column::Broadcast {
                input: Box::new(table),
                low: low as usize,
                len: 1usize << a_vars,
            })));
        }
        let table = table.resident()?;
        let mask = (1usize << b_vars) - 1;
        let mut full = public_zeros(1usize << a_vars)?;
        for (i, slot) in full.iter_mut().enumerate() {
            *slot = table[(i >> low) & mask];
        }
        Ok(Read::Public(arena.public(full)))
    }

    fn channels(&self, node: &Node, streaming: bool) -> Result<Column, StageError> {
        if !streaming {
            return Ok(self.dense_channels(node)?.into());
        }
        let Op::ConstMul { c } = node.op else {
            return internal("a channel constant is a ConstMul's");
        };
        let (constant, shape) = constant_and_shape(self.graph, node, c)?;
        let constants = constant.to_vec();
        let width = 1usize << shape.last().map_or(0, |&n| axis_vars(n));
        Column::generated(
            true,
            1usize << Self::vars_of(shape),
            array::<i64>(constants.capacity())?,
            move |i| {
                Ok(constants
                    .get(i % width)
                    .copied()
                    .map_or(Goldilocks::ZERO, Goldilocks::from_i64))
            },
        )
    }

    fn shifted(&self, node: &Node, streaming: bool) -> Result<Column, StageError> {
        if !streaming {
            return Ok(self.dense_shifted(node)?.into());
        }
        let (constant, shape) = const_add(self.graph, node.id)?;
        let constants = constant.to_vec();
        let shape = shape.to_vec();
        Column::generated(
            true,
            1usize << Self::vars_of(&shape),
            add(
                array::<i64>(constants.capacity())?,
                array::<u32>(shape.capacity())?,
            )?,
            move |i| {
                let mut rest = i;
                let channel_bits = shape.last().map_or(0, |&n| axis_vars(n));
                let channel = i & ((1usize << channel_bits) - 1);
                for &n in shape.iter().rev() {
                    let bits = axis_vars(n);
                    if (rest & ((1usize << bits) - 1)) >= n as usize {
                        return Ok(Goldilocks::ZERO);
                    }
                    rest >>= bits;
                }
                Ok(Goldilocks::from_i64(if constants.len() == 1 {
                    constants[0]
                } else {
                    constants[channel]
                }))
            },
        )
    }

    /// The constant per channel of the `ConstMul` `node`, broadcast over the other axes of its
    /// output, 0 at padded channels: `c̃(x_c)` as a public table of every variable.
    fn dense_channels(&self, node: &Node) -> Result<Vec<Goldilocks>, StageError> {
        let Op::ConstMul { c } = node.op else {
            return internal("a constant per channel is a ConstMul's");
        };
        let (constant, shape) = constant_and_shape(self.graph, node, c)?;
        let low = shape.last().map_or(0, |&len| axis_vars(len)) as usize;
        let channels = channel_table(constant, low)?;
        let mut table = public_zeros(1usize << Self::vars_of(shape))?;
        for (i, slot) in table.iter_mut().enumerate() {
            *slot = channels[i & ((1usize << low) - 1)];
        }
        Ok(table)
    }

    /// The table of `ψ·c` of the `ConstAdd` `node`: its constant at each real position of its
    /// output, 0 at padded ones (RFC-0072 'Padded positions'), whose extension is the shift `ĉ`.
    fn dense_shifted(&self, node: &Node) -> Result<Vec<Goldilocks>, StageError> {
        let (constant, shape) = const_add(self.graph, node.id)?;
        let count: usize = shape.iter().map(|&len| len as usize).product();
        let channels = shape.last().map_or(1, |&len| len as usize).max(1);
        let values: Vec<i64> = (0..count)
            .map(|i| match constant {
                [scalar] => *scalar,
                _ => constant[i % channels],
            })
            .collect();
        let mut table = public_zeros(1usize << Self::vars_of(shape))?;
        pad(&values, shape, &mut table)?;
        Ok(table)
    }

    /// `Mul`'s second operand as its relation reads it: a broadcast window of the product's
    /// variables, or, for one channel per layer, laid out over the rows.
    fn broadcast(
        &self,
        arena: &mut Arena,
        b: Column,
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
                if arena.streaming {
                    return Ok(Read::Spread(arena.secret(Column::Spread {
                        input: Box::new(b),
                        block,
                        rows,
                        len: layers * rows * block,
                    })));
                }
                let b = b.resident()?;
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
        streaming: bool,
    ) -> Result<(Column, Column), StageError> {
        let Op::Rotate { rope } = node.op else {
            return internal("a Rotate names its RoPE table");
        };
        let spec = self
            .tables
            .iter()
            .find(|spec| spec.id == rope)
            .ok_or(StageError::Internal {
                rule: "a Rotate's table is the graph's",
            })?;
        let table = RopeTable::from_spec(spec).map_err(|_| StageError::Internal {
            rule: "a Rotate's table regenerates",
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
        if streaming {
            if !self.public.contains(node.inputs[1]) {
                return internal("a streamed Rotate uses public positions");
            }
            let table = std::sync::Arc::new(table);
            let slots = self.graph.slots.clone().map(std::sync::Arc::new);
            let column = |sine| Column::Rope {
                table: table.clone(),
                positions: node.inputs[1],
                slots: slots.clone(),
                bit,
                head_vars,
                row_vars,
                rows: rows as usize,
                sine,
                len,
            };
            return Ok((column(false), column(true)));
        }
        let positions = self
            .witness
            .values(node.inputs[1])
            .ok_or(StageError::Internal {
                rule: "the witness holds the positions",
            })?;
        let (mut cos, mut sin) = (vec![Goldilocks::ZERO; len], vec![Goldilocks::ZERO; len]);
        for k in 0..len {
            let frequency = k & ((1 << bit) - 1);
            let row = (k >> (bit + head_vars)) & ((1 << row_vars) - 1);
            if row >= rows as usize
                || self
                    .graph
                    .slots
                    .as_ref()
                    .is_some_and(|slots| !slots.is_real_row(row))
            {
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
        Ok((cos.into(), sin.into()))
    }

    /// The equality zero-check `0 = Σ_x eq(z, x)·(Ỹ(x) − f(x))` of `node`'s committed output
    /// (RFC-0037 'The committed first checkpoint', RFC-0072 'A committed output'), its evaluations
    /// `Y`'s then those of `f`'s inputs that are not public: a public input is read as `Public`.
    /// A `ConstAdd`'s `f = x̃ + ĉ` reads the table of `ψ·c`, and a per-channel `ConstMul`'s
    /// `f = c̃(x_c)·x̃` its constant broadcast over the rows, both public.
    fn equality(
        &self,
        arena: &mut Arena,
        instance: &Planned,
        node: &Node,
        operand: &dyn Fn(usize) -> Result<Column, StageError>,
        stacked: bool,
    ) -> Result<Plan, StageError> {
        let layers = &self.schedule.layers;
        let lambda = if stacked { layers.vars as usize } else { 0 };
        let output = &self.graph.tensors[node.outputs[0].0 as usize].shape;
        let shape = ElementwiseShape::elementwise(Self::vars_of(output) as usize + lambda);
        let one = GoldilocksExt2::ONE;
        let y = Read::Operand(arena.secret(operand(0)?));
        // `f`'s claimed inputs take the instance's claims after `Y`'s, in `f`'s order.
        let mut next = 1;
        let mut read = |arena: &mut Arena, i: usize| -> Result<Read, StageError> {
            Ok(
                match self.input(instance, node, i, operand, &mut next, arena.streaming)? {
                    Operand::Public(table) => Read::Public(arena.public(table)),
                    Operand::Claimed(table) => Read::Operand(arena.secret(table)),
                },
            )
        };
        let (inputs, terms) = match node.op {
            Op::ConstMul { c } => {
                let x = read(arena, 0)?;
                let (constant, _) = constant_and_shape(self.graph, node, c)?;
                if constant.len() > 1 {
                    if stacked {
                        return internal(
                            "a ConstMul by a constant per channel is outside the layers",
                        );
                    }
                    let c = Read::Public(arena.public(self.channels(node, arena.streaming)?));
                    (
                        vec![y, x, c],
                        vec![Term::new(one, vec![0]), Term::new(-one, vec![2, 1])],
                    )
                } else {
                    self.scaled(arena, instance, y, x, output)?
                }
            }
            Op::ConstAdd { .. } => {
                if stacked {
                    return internal("a ConstAdd is outside the layers");
                }
                let x = read(arena, 0)?;
                let shift = Read::Public(arena.public(self.shifted(node, arena.streaming)?));
                (
                    vec![y, x, shift],
                    vec![
                        Term::new(one, vec![0]),
                        Term::new(-one, vec![1]),
                        Term::new(-one, vec![2]),
                    ],
                )
            }
            Op::Add | Op::Sub => {
                let a = read(arena, 0)?;
                let b = read(arena, 1)?;
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
                let a_shape = &self.graph.tensors[node.inputs[0].0 as usize].shape;
                let b_shape = &self.graph.tensors[node.inputs[1].0 as usize].shape;
                let a = read(arena, 0)?;
                let b = self.input(instance, node, 1, operand, &mut next, arena.streaming)?;
                let b = self.mul_operand(arena, b, a_shape, b_shape, broadcast, stacked)?;
                (
                    vec![y, a, b],
                    vec![Term::new(one, vec![0]), Term::new(-one, vec![1, 2])],
                )
            }
            _ => {
                return internal(
                    "a committed output's zero-check is of a ConstMul, ConstAdd, Add, Sub or Mul",
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

    /// The inputs and terms of `Y − c·X` for a `ConstMul` by one constant: `c` a coefficient, or
    /// at a template position whose layers' constants differ, each layer's own as a public table
    /// constant within each layer.
    fn scaled(
        &self,
        arena: &mut Arena,
        instance: &Planned,
        y: Read,
        x: Read,
        output: &[u32],
    ) -> Result<(Vec<Read>, Vec<Term<GoldilocksExt2>>), StageError> {
        let layers = &self.schedule.layers;
        let one = GoldilocksExt2::ONE;
        let constant = |layer: u32| -> Result<i64, StageError> {
            let id = match instance.node {
                NodeRef::Node(id) => id,
                NodeRef::Position(position) => layers.node(position, layer),
            };
            match self.graph.nodes[id.0 as usize].op {
                Op::ConstMul { c } => self
                    .graph
                    .constants
                    .get(c.0 as usize)
                    .and_then(|v| match v[..] {
                        [scalar] => Some(scalar),
                        _ => None,
                    })
                    .ok_or(StageError::Internal {
                        rule: "a committed ConstMul's constant is one scalar in every layer",
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
            return Ok((
                vec![y, x],
                vec![Term::new(one, vec![0]), Term::new(-c, vec![1])],
            ));
        }
        let block = 1usize << Self::vars_of(output);
        let values = (0..layers.count)
            .map(|l| constant(l).map(Goldilocks::from_i64))
            .collect::<Result<Vec<_>, _>>()?;
        let table = Column::generated(
            arena.streaming,
            block << layers.vars,
            array::<Goldilocks>(values.capacity())?,
            move |i| Ok(values.get(i / block).copied().unwrap_or(Goldilocks::ZERO)),
        )?;
        let c = Read::Public(arena.public(table));
        Ok((
            vec![y, x, c],
            vec![Term::new(one, vec![0]), Term::new(-one, vec![2, 1])],
        ))
    }
}

/// The axis a selection's node selects along and its rows `idx_j`, `j < len`: a `Slice`'s
/// `start + j` along its axis, a `GatherPublic`'s constant along the outermost.
///
/// Every row is checked against the input's axis, `idx_j < n_a`, so that no padded row of `x` is
/// selected (RFC-0072 'The selection relation' 'Padded rows'), as graph validation also
/// requires: the kernel bounds rows only by `2^{⌈log2 n_a⌉}`.
fn selected(graph: &tessera_ir::Graph, node: &Node) -> Result<(usize, Vec<usize>), StageError> {
    let (axis, index) = selected_rows(graph, node)?;
    let n_a = node
        .inputs
        .first()
        .and_then(|t| graph.tensors.get(t.0 as usize))
        .and_then(|x| x.shape.get(axis))
        .ok_or(StageError::Internal {
            rule: "a selection reads an axis of its input",
        })?;
    if index.iter().any(|&row| row >= *n_a as usize) {
        return internal("a selection's rows are rows of its input's axis");
    }
    Ok((axis, index))
}

/// The axis and rows a `Slice` or a `GatherPublic` takes, unchecked.
fn selected_rows(
    graph: &tessera_ir::Graph,
    node: &Node,
) -> Result<(usize, Vec<usize>), StageError> {
    match node.op {
        Op::Slice { axis, start, len } => Ok((
            usize::from(axis),
            (start as usize..start as usize + len as usize).collect(),
        )),
        Op::GatherPublic { index } => {
            let rows = graph
                .constants
                .get(index.0 as usize)
                .ok_or(StageError::Internal {
                    rule: "a gather's indices are a constant of the graph",
                })?;
            let index = rows
                .iter()
                .map(|&row| usize::try_from(row))
                .collect::<Result<_, _>>()
                .map_err(|_| StageError::Internal {
                    rule: "a gather's indices are rows of its input",
                })?;
            Ok((0, index))
        }
        _ => internal("a selection is a Slice's or a GatherPublic's"),
    }
}

/// RFC-0066's multilinear polynomial of `f` on its inputs `0, 1, 2` as the terms of an
/// `Elementwise` relation: `Xor` `a + b − 2ab`, `Xor3` `a + b + c − 2ab − 2ac − 2bc + 4abc`, `And`
/// `ab`, `Ch` `ef + g − eg` and `Maj` `ab + ac + bc − 2abc`.
fn bit_terms(f: BoolFn) -> Vec<Term<GoldilocksExt2>> {
    let c = |v: i64| GoldilocksExt2::from(Goldilocks::from_i64(v));
    let term = |weight: i64, factors: &[usize]| Term::new(c(weight), factors.to_vec());
    match f {
        BoolFn::Xor => vec![term(1, &[0]), term(1, &[1]), term(-2, &[0, 1])],
        BoolFn::Xor3 => vec![
            term(1, &[0]),
            term(1, &[1]),
            term(1, &[2]),
            term(-2, &[0, 1]),
            term(-2, &[0, 2]),
            term(-2, &[1, 2]),
            term(4, &[0, 1, 2]),
        ],
        BoolFn::And => vec![term(1, &[0, 1])],
        BoolFn::Ch => vec![term(1, &[0, 1]), term(1, &[2]), term(-1, &[0, 2])],
        BoolFn::Maj => vec![
            term(1, &[0, 1]),
            term(1, &[0, 2]),
            term(1, &[1, 2]),
            term(-2, &[0, 1, 2]),
        ],
    }
}

fn internal<T>(rule: &'static str) -> Result<T, StageError> {
    Err(StageError::Internal { rule })
}

#[cfg(test)]
mod slot_tests {
    use super::*;
    use tessera_codec::{DecodeLimits, envelope, schema};
    use tessera_compiler::{PublicModel, derive_schedule, lower_batch, positions_values};
    use tessera_ir::{IntegerModel, TensorId};

    struct Positions {
        tensor: TensorId,
        values: Vec<i64>,
    }
    impl WitnessSource for Positions {
        fn values(&self, tensor: TensorId) -> Option<&[i64]> {
            (tensor == self.tensor).then_some(&self.values)
        }
    }

    #[test]
    fn rope_public_tables_are_zero_on_inter_slot_and_axis_padding() {
        let model: IntegerModel = envelope::open(
            include_bytes!("../../../../spec/vectors/models/tiny-2-minimal.d"),
            &schema::INTEGER_MODEL,
            DecodeLimits::INTEGER_MODEL,
        )
        .unwrap();
        let lowering = lower_batch(PublicModel::of(&model), &[2, 3, 1]).unwrap();
        let schedule = derive_schedule(&lowering).unwrap();
        let node = lowering
            .graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Rotate { .. }))
            .unwrap();
        let positions = Positions {
            tensor: node.inputs[1],
            values: positions_values(&lowering),
        };
        let prover = StageProver::new(&lowering, &schedule, &positions);
        let shape = &lowering.graph.tensors[node.inputs[0].0 as usize].shape;
        let mut resident = None;
        for streaming in [false, true] {
            let (cos, sin) = prover.rope(node, shape, false, streaming).unwrap();
            let read = |column: &Column| {
                (0..column.len())
                    .map(|i| column.value(&positions, i).unwrap())
                    .collect::<Vec<_>>()
            };
            let (cos, sin) = (read(&cos), read(&sin));
            if let Some((ref a, ref b)) = resident {
                assert_eq!((&cos, &sin), (a, b));
            } else {
                resident = Some((cos.clone(), sin.clone()));
            }
            let rows = shape[0].next_power_of_two() as usize;
            let per_row = cos.len() / rows;
            assert!(cos[..per_row].iter().any(|&c| c != Goldilocks::ZERO));
            // Real rows are 0,1; 4,5,6; and 8. Every other row must have a zero kernel.
            for row in [2, 3, 7, 9, 10, 11, 12, 13, 14, 15] {
                assert!(
                    cos[row * per_row..(row + 1) * per_row]
                        .iter()
                        .all(|&c| c == Goldilocks::ZERO)
                );
                assert!(
                    sin[row * per_row..(row + 1) * per_row]
                        .iter()
                        .all(|&c| c == Goldilocks::ZERO)
                );
            }
        }
    }
}
