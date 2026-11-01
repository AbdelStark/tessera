// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0055 regenerated stage columns and public relation geometry.

use std::time::Instant;
use tessera_compiler::schedule::{Layers, Relation};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_poly::PolyField;
use tessera_sumcheck::{
    PreparedWindow, Protocol, Table, Term, WindowError, WindowInstance, WindowPass, WindowSource,
    WindowTable,
};

use super::columns::Column;
use super::relations::{Arena, Plan, Read};
use super::source_memory::{add, array};
use super::{NativeProver, StageError, StageProver, WitnessSource, eq_vertex};

type F = GoldilocksExt2;

#[derive(Clone, Copy)]
struct ColumnRef {
    public: bool,
    index: usize,
}
#[derive(Clone, Copy)]
enum Coordinate {
    Free(usize),
    Fixed(F),
}

enum Factor {
    Column {
        column: ColumnRef,
        coordinates: Vec<Coordinate>,
    },
    Public {
        at: Box<dyn Fn(usize) -> F + Send + Sync>,
        heap: u64,
    },
    Rotate {
        cos: ColumnRef,
        sin: ColumnRef,
        bit: usize,
        rest: Vec<F>,
        at_bit: F,
    },
    Native {
        tag: u32,
        table: usize,
    },
}

fn internal<T>(rule: &'static str) -> Result<T, StageError> {
    Err(StageError::Internal { rule })
}
fn map_error(error: WindowError) -> StageError {
    match error {
        WindowError::Allocation => StageError::Memory,
        WindowError::Shape(rule) => StageError::Internal { rule },
        WindowError::Source => StageError::Internal {
            rule: "a regenerated stage supplies its declared columns",
        },
    }
}

impl Arena {
    fn column(&self, column: ColumnRef) -> &Column {
        if column.public {
            &self.public[column.index]
        } else {
            &self.secret[column.index]
        }
    }
}

impl Factor {
    fn public(
        owned: u64,
        at: impl Fn(usize) -> F + Send + Sync + 'static,
    ) -> Result<Self, StageError> {
        let heap = add(size_of_val(&at) as u64, owned)?;
        Ok(Self::Public {
            at: Box::new(at),
            heap,
        })
    }
    fn heap_bytes(&self) -> Result<u64, StageError> {
        match self {
            Self::Column { coordinates, .. } => array::<Coordinate>(coordinates.capacity()),
            Self::Public { heap, .. } => Ok(*heap),
            Self::Rotate { rest, .. } => array::<F>(rest.capacity()),
            Self::Native { .. } => Ok(0),
        }
    }
    fn read(read: &Read, arena: &Arena) -> Self {
        let (column, low) = match *read {
            Read::Operand(index) | Read::Spread(index) => (
                ColumnRef {
                    public: false,
                    index,
                },
                0,
            ),
            Read::Broadcast { table, low_vars } => (
                ColumnRef {
                    public: false,
                    index: table,
                },
                low_vars,
            ),
            Read::Public(index) => (
                ColumnRef {
                    public: true,
                    index,
                },
                0,
            ),
        };
        let vars = arena.column(column).len().trailing_zeros() as usize;
        Self::Column {
            column,
            coordinates: (0..vars).map(|j| Coordinate::Free(j + low)).collect(),
        }
    }

    fn value<W: WitnessSource + ?Sized>(
        &self,
        arena: &Arena,
        witness: &W,
        index: usize,
    ) -> Result<F, StageError> {
        match self {
            Self::Column {
                column,
                coordinates,
            } => {
                let mut base = 0usize;
                let mut fixed_count = 0;
                for (bit, coord) in coordinates.iter().enumerate() {
                    match *coord {
                        Coordinate::Free(j) => base |= ((index >> j) & 1) << bit,
                        Coordinate::Fixed(_) => fixed_count += 1,
                    }
                }
                let mut sum = F::ZERO;
                for vertex in 0..1usize << fixed_count {
                    let mut at = base;
                    let mut weight = F::ONE;
                    let mut j = 0;
                    for (bit, coordinate) in coordinates.iter().enumerate() {
                        if let Coordinate::Fixed(r) = coordinate {
                            let b = vertex >> j & 1;
                            at |= b << bit;
                            weight *= if b == 0 { F::ONE - *r } else { *r };
                            j += 1;
                        }
                    }
                    sum += weight.mul_by_base(arena.column(*column).value(witness, at)?);
                }
                Ok(sum)
            }
            Self::Public { at, .. } => Ok(at(index)),
            Self::Rotate {
                cos,
                sin,
                bit,
                rest,
                at_bit,
            } => {
                let low = (1usize << bit) - 1;
                let pair = ((index >> (bit + 1)) << bit) | (index & low);
                let c = arena.column(*cos).value(witness, pair)?;
                let s = arena.column(*sin).value(witness, pair)?;
                let kernel = if index >> bit & 1 == 0 {
                    (F::ONE - *at_bit).mul_by_base(c) + at_bit.mul_by_base(s)
                } else {
                    at_bit.mul_by_base(c) + (*at_bit - F::ONE).mul_by_base(s)
                };
                Ok(eq_vertex(rest, pair as u64) * kernel)
            }
            Self::Native { .. } => {
                internal("the sumcheck engine reads native opaque resident tables")
            }
        }
    }
}

pub(super) struct PreparedStage {
    pub(super) index: u16,
    cut: usize,
    arena: Arena,
    plans: Vec<Plan>,
    points: Vec<Vec<F>>,
    graph_indices: Vec<Option<usize>>,
    factors: Vec<Factor>,
    pub(super) prepared: PreparedWindow<F>,
}

struct Source<'a, R, W> {
    arena: &'a Arena,
    factors: &'a [Factor],
    native: &'a dyn NativeProver,
    regenerate: &'a mut R,
    witness: Option<W>,
    started: Option<Instant>,
}
impl<R, W> WindowSource<F> for Source<'_, R, W>
where
    R: FnMut() -> Result<W, StageError>,
    W: WitnessSource,
{
    fn regenerate(&mut self, _pass: WindowPass) -> Result<(), WindowError> {
        self.started = Some(Instant::now());
        self.witness = None;
        self.witness = Some((self.regenerate)().map_err(|e| {
            if matches!(e, StageError::Memory) {
                WindowError::Allocation
            } else {
                WindowError::Source
            }
        })?);
        Ok(())
    }
    fn finish_pass(&mut self, pass: WindowPass) -> Result<(), WindowError> {
        let elapsed = self.started.take().ok_or(WindowError::Source)?.elapsed();
        let label = match pass {
            WindowPass::Prepare => tessera_codec::metrics::Pass::P3,
            WindowPass::Fold(_) => tessera_codec::metrics::Pass::P3b,
        };
        tessera_codec::metrics::pass(label, elapsed);
        Ok(())
    }
    fn resident(&self, id: usize) -> Option<Table<'_, F>> {
        match self.factors.get(id)? {
            Factor::Native { tag, table } => self
                .native
                .instance(*tag)
                .ok()?
                .tables()
                .get(*table)
                .copied(),
            _ => None,
        }
    }
    fn read(&self, id: usize, start: usize, out: &mut [F]) -> Result<(), WindowError> {
        let factor = self.factors.get(id).ok_or(WindowError::Source)?;
        let witness = self.witness.as_ref().ok_or(WindowError::Source)?;
        for (offset, value) in out.iter_mut().enumerate() {
            *value = factor
                .value(self.arena, witness, start + offset)
                .map_err(|e| {
                    if matches!(e, StageError::Memory) {
                        WindowError::Allocation
                    } else {
                        WindowError::Source
                    }
                })?;
        }
        Ok(())
    }
}

fn push(factors: &mut Vec<Factor>, factor: Factor, public: bool) -> WindowTable {
    let id = factors.len();
    factors.push(factor);
    WindowTable { id, public }
}
fn column(index: usize) -> ColumnRef {
    ColumnRef {
        public: false,
        index,
    }
}
fn fixed(point: &[F]) -> impl Iterator<Item = Coordinate> + '_ {
    point.iter().copied().map(Coordinate::Fixed)
}
fn free(start: usize, count: usize) -> impl Iterator<Item = Coordinate> {
    (start..start + count).map(Coordinate::Free)
}
fn eq_factor(point: Vec<F>, low: usize, zero_bits: usize) -> Result<Factor, StageError> {
    Factor::public(array::<F>(point.capacity())?, move |index| {
        if index & ((1usize << zero_bits) - 1) != 0 {
            F::ZERO
        } else {
            eq_vertex(&point, (index >> low) as u64)
        }
    })
}

impl Plan {
    fn window(
        &self,
        arena: &Arena,
        point: &[F],
        factors: &mut Vec<Factor>,
    ) -> Result<WindowInstance<F>, StageError> {
        let mut tables = Vec::new();
        let mut terms = Vec::new();
        let mut eq = Vec::new();
        let vars;
        let mut windowed = true;
        match self {
            Self::Elementwise {
                inputs,
                terms: own,
                shape,
                point_from,
            } => {
                vars = shape.vars;
                let point = point.get(*point_from..).ok_or(StageError::Internal {
                    rule: "a windowed row check projects its zero point",
                })?;
                if point.len() != vars - shape.summed_vars {
                    return internal("a windowed EQ point has its target variables");
                }
                eq = vec![None; shape.summed_vars];
                eq.extend(point.iter().copied().map(Some));
                for input in inputs {
                    tables.push(push(
                        factors,
                        Factor::read(input, arena),
                        matches!(input, Read::Public(_)),
                    ));
                }
                terms = own.clone();
            }
            Self::MatMulWeight { w, x, shape } => {
                vars = shape.in_vars + shape.layer_vars;
                windowed = false;
                let (out, rest) = point.split_at(shape.out_vars);
                let (token, layer) = rest.split_at(shape.token_vars);
                if layer.len() != shape.layer_vars {
                    return internal("a windowed weight matmul has its public point");
                }
                for (input, middle) in [(*w, out), (*x, token)] {
                    let coordinates = free(0, shape.in_vars)
                        .chain(fixed(middle))
                        .chain(free(shape.in_vars, shape.layer_vars))
                        .collect();
                    tables.push(push(
                        factors,
                        Factor::Column {
                            column: column(input),
                            coordinates,
                        },
                        false,
                    ));
                }
                eq = vec![None; shape.in_vars];
                eq.extend(layer.iter().copied().map(Some));
                terms.push(Term::new(F::ONE, vec![0, 1]));
            }
            Self::MatMulAct {
                a,
                b,
                layout,
                shape,
            } => {
                vars = shape.inner_vars + shape.stack_vars;
                windowed = false;
                let (cols, rest) = point.split_at(shape.col_vars);
                let (rows, stack) = rest.split_at(shape.row_vars);
                if stack.len() != shape.stack_vars {
                    return internal("a windowed activation matmul has its public point");
                }
                let left = free(0, shape.inner_vars)
                    .chain(fixed(rows))
                    .chain(free(shape.inner_vars, shape.stack_vars))
                    .collect();
                let right = match layout {
                    tessera_sumcheck::RightLayout::Direct => fixed(cols)
                        .chain(free(0, shape.inner_vars))
                        .chain(free(shape.inner_vars, shape.stack_vars))
                        .collect(),
                    tessera_sumcheck::RightLayout::Transposed => free(0, shape.inner_vars)
                        .chain(fixed(cols))
                        .chain(free(shape.inner_vars, shape.stack_vars))
                        .collect(),
                };
                tables.push(push(
                    factors,
                    Factor::Column {
                        column: column(*a),
                        coordinates: left,
                    },
                    false,
                ));
                tables.push(push(
                    factors,
                    Factor::Column {
                        column: column(*b),
                        coordinates: right,
                    },
                    false,
                ));
                eq = vec![None; shape.inner_vars];
                eq.extend(stack.iter().copied().map(Some));
                terms.push(Term::new(F::ONE, vec![0, 1]));
            }
            Self::Rotate {
                x,
                cos,
                sin,
                bit,
                vars: n,
            } => {
                vars = *n;
                if point.len() != vars {
                    return internal("a windowed Rotate point has its tensor variables");
                }
                tables.push(push(
                    factors,
                    Factor::Rotate {
                        cos: ColumnRef {
                            public: true,
                            index: *cos,
                        },
                        sin: ColumnRef {
                            public: true,
                            index: *sin,
                        },
                        bit: *bit,
                        rest: point
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| j != bit)
                            .map(|(_, &r)| r)
                            .collect(),
                        at_bit: point[*bit],
                    },
                    true,
                ));
                tables.push(push(
                    factors,
                    Factor::read(&Read::Operand(*x), arena),
                    false,
                ));
                terms.push(Term::new(F::ONE, vec![0, 1]));
            }
            Self::Selection {
                x,
                low,
                axis,
                high,
                rows,
                members,
            } => {
                vars = *axis;
                if point.len() != low + rows + high {
                    return internal("a windowed selection point has its output variables");
                }
                let (lo, rest) = point.split_at(*low);
                let (selected, hi) = rest.split_at(*rows);
                let members = members.clone();
                let selected = selected.to_vec();
                let owned = members.iter().try_fold(
                    array::<(F, Vec<usize>)>(members.capacity())?,
                    |n, (_, v)| add(n, array::<usize>(v.capacity())?),
                )?;
                let owned = add(owned, array::<F>(selected.capacity())?)?;
                tables.push(push(
                    factors,
                    Factor::public(owned, move |index| {
                        members.iter().fold(F::ZERO, |sum, (coefficient, indices)| {
                            sum + *coefficient
                                * indices
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, i)| **i == index)
                                    .fold(F::ZERO, |s, (j, _)| s + eq_vertex(&selected, j as u64))
                        })
                    })?,
                    true,
                ));
                let coordinates = fixed(lo).chain(free(0, *axis)).chain(fixed(hi)).collect();
                tables.push(push(
                    factors,
                    Factor::Column {
                        column: column(*x),
                        coordinates,
                    },
                    false,
                ));
                terms.push(Term::new(F::ONE, vec![0, 1]));
            }
            Self::Recomposition {
                x,
                b,
                h,
                low,
                high,
                rows,
                point_from,
            } => {
                let kb = u32::from(*low).next_power_of_two().trailing_zeros() as usize;
                let kh = u32::from(*high).next_power_of_two().trailing_zeros() as usize;
                let k = kb.max(kh);
                vars = k + rows;
                let point = point.get(*point_from..).ok_or(StageError::Internal {
                    rule: "a windowed recomposition projects its row point",
                })?;
                if point.len() != *rows {
                    return internal("a recomposition EQ has its row coordinates");
                }
                tables.push(push(factors, eq_factor(point.to_vec(), k, 0)?, true));
                let mut whole = Factor::read(x, arena);
                if let Factor::Column { coordinates, .. } = &mut whole {
                    for coordinate in coordinates {
                        if let Coordinate::Free(j) = coordinate {
                            *j += k;
                        }
                    }
                }
                tables.push(push(factors, whole, matches!(x, Read::Public(_))));
                let scale = F::from(Goldilocks::from_u64_reduced(1u64 << k))
                    .inverse()
                    .ok_or(StageError::Memory)?;
                terms.push(Term::new(scale, vec![0, 1]));
                for (input, count, bits, weight) in [
                    (Some(*b), *low, kb, -F::ONE),
                    (
                        *h,
                        *high,
                        kh,
                        -F::from(Goldilocks::from_u64_reduced(1u64 << low)),
                    ),
                ] {
                    let Some(input) = input else {
                        continue;
                    };
                    let pad = k - bits;
                    let at = tables.len();
                    tables.push(push(factors, eq_factor(point.to_vec(), k, pad)?, true));
                    tables.push(push(
                        factors,
                        Factor::public(0, move |index| {
                            let i = index & ((1usize << k) - 1);
                            let bit = i >> pad;
                            if i & ((1usize << pad) - 1) == 0 && bit < usize::from(count) {
                                F::from(Goldilocks::from_u64_reduced(1u64 << bit))
                            } else {
                                F::ZERO
                            }
                        })?,
                        true,
                    ));
                    tables.push(push(
                        factors,
                        Factor::Column {
                            column: column(input),
                            coordinates: free(pad, bits + rows).collect(),
                        },
                        false,
                    ));
                    terms.push(Term::new(weight, vec![at, at + 1, at + 2]));
                }
            }
        }
        if eq.is_empty() {
            eq = vec![None; vars];
        }
        Ok(WindowInstance {
            vars,
            tables,
            terms,
            eq,
            windowed,
        })
    }
}

impl<W: WitnessSource + ?Sized> StageProver<'_, W> {
    pub(super) fn prepare_window<R, V>(
        &self,
        k: u16,
        cut: usize,
        native: &dyn NativeProver,
        regenerate: &mut R,
    ) -> Result<PreparedStage, StageError>
    where
        R: FnMut() -> Result<V, StageError>,
        V: WitnessSource,
    {
        let stage = self
            .schedule
            .stages
            .get(usize::from(k))
            .ok_or(StageError::Internal {
                rule: "a windowed stage is compiled",
            })?;
        let mut arena = Arena {
            streaming: true,
            ..Arena::default()
        };
        let mut plans = Vec::new();
        let mut points = Vec::new();
        let mut graph_indices = Vec::new();
        let mut factors = Vec::new();
        let mut specs = Vec::new();
        let mut degree = 1;
        for instance in &stage.instances {
            if let Relation::Native {
                tag,
                degree: d,
                evaluations,
                ..
            } = instance.relation
            {
                graph_indices.push(None);
                let own = native.instance(tag)?;
                if own.tables().iter().filter(|t| !t.is_public()).count() != evaluations
                    || own
                        .tables()
                        .iter()
                        .any(|t| t.len() != 1usize << instance.vars)
                {
                    return internal("a native resident instance fits its compiled shape");
                }
                let tables = own
                    .tables()
                    .iter()
                    .enumerate()
                    .map(|(table, t)| {
                        push(&mut factors, Factor::Native { tag, table }, t.is_public())
                    })
                    .collect();
                let terms = own.terms().map_or_else(
                    || vec![Term::new(F::ONE, (0..own.tables().len()).collect())],
                    <[_]>::to_vec,
                );
                specs.push(WindowInstance {
                    vars: instance.vars as usize,
                    tables,
                    terms,
                    eq: vec![None; instance.vars as usize],
                    windowed: false,
                });
                degree = degree.max(d);
            } else {
                graph_indices.push(Some(plans.len()));
                let plan = self.plan(stage, instance, &mut arena)?;
                let mut point = self.point(&instance.point)?;
                if let Layers::Eq(rho) = &instance.layers {
                    point.extend(self.point(rho)?);
                }
                let spec = plan.window(&arena, &point, &mut factors)?;
                let charged = plan.memory_shape(instance.vars as usize);
                if charged.tables != spec.tables.len()
                    || charged.delta
                        != spec
                            .terms
                            .iter()
                            .map(|t| t.factors().len())
                            .max()
                            .unwrap_or(1)
                    || charged.windowable != spec.windowed
                {
                    return internal("the pre-P1 stage geometry charges every regenerated column");
                }
                let d = spec
                    .terms
                    .iter()
                    .map(|t| t.factors().len())
                    .max()
                    .unwrap_or(1)
                    + usize::from(spec.eq.iter().any(Option::is_some));
                degree = degree.max(d);
                if spec.vars != instance.vars as usize {
                    return internal("a regenerated instance has its compiled variable count");
                }
                specs.push(spec);
                plans.push(plan);
                points.push(point);
            }
        }
        let mut source = Source {
            arena: &arena,
            factors: &factors,
            native,
            regenerate,
            witness: None,
            started: None,
        };
        let prepared = PreparedWindow::new(&mut source, Protocol::One, degree, specs, cut)
            .map_err(map_error)?;
        Ok(PreparedStage {
            index: k,
            cut,
            arena,
            plans,
            points,
            graph_indices,
            factors,
            prepared,
        })
    }
}

impl Plan {
    fn claim_points(
        &self,
        arena: &Arena,
        fixed_point: &[F],
        point: &[F],
    ) -> Result<Vec<Vec<F>>, StageError> {
        Ok(match self {
            Self::Elementwise { inputs, .. } => inputs
                .iter()
                .filter_map(|input| match *input {
                    Read::Public(_) => None,
                    Read::Operand(_) | Read::Spread(_) => Some(point.to_vec()),
                    Read::Broadcast { table, low_vars } => Some(
                        point[low_vars
                            ..low_vars + arena.secret[table].len().trailing_zeros() as usize]
                            .to_vec(),
                    ),
                })
                .collect(),
            Self::Rotate { .. } => vec![point.to_vec()],
            Self::Selection { low, rows, .. } => {
                vec![[&fixed_point[..*low], point, &fixed_point[low + rows..]].concat()]
            }
            Self::MatMulWeight { shape, .. } => {
                let (inner, layer) = point.split_at(shape.in_vars);
                let (out, rest) = fixed_point.split_at(shape.out_vars);
                let (token, _) = rest.split_at(shape.token_vars);
                vec![[inner, out, layer].concat(), [inner, token, layer].concat()]
            }
            Self::MatMulAct { shape, layout, .. } => {
                let (inner, stack) = point.split_at(shape.inner_vars);
                let (cols, rest) = fixed_point.split_at(shape.col_vars);
                let (rows, _) = rest.split_at(shape.row_vars);
                let right = match layout {
                    tessera_sumcheck::RightLayout::Direct => [cols, inner, stack].concat(),
                    tessera_sumcheck::RightLayout::Transposed => [inner, cols, stack].concat(),
                };
                vec![[inner, rows, stack].concat(), right]
            }
            Self::Recomposition {
                x, low, high, h, ..
            } => {
                let kb = u32::from(*low).next_power_of_two().trailing_zeros() as usize;
                let kh = u32::from(*high).next_power_of_two().trailing_zeros() as usize;
                let k = kb.max(kh);
                let mut points = Vec::new();
                if !matches!(x, Read::Public(_)) {
                    points.push(point[k..].to_vec());
                }
                points.push(point[k - kb..].to_vec());
                if h.is_some() {
                    points.push(point[k - kh..].to_vec());
                }
                points
            }
        })
    }
}

impl<W: WitnessSource + ?Sized> StageProver<'_, W> {
    /// Prove a stage with RFC-0055 windows and prepare the following stage's grid before this
    /// stage's splits. `cut` and `next_cut` are public global-round cuts from the memory plan;
    /// zero keeps that stage's resident oracle. Regeneration is required at every P3/P3b.
    /// Native instances must supply resident tables, as charged by the memory projection;
    /// computed round states are not supported by this engine. The CPU driver selects
    /// `ModelNativeProver::new_literal` for its windowed route.
    ///
    /// # Errors
    /// The ordinary stage errors, or a regenerated witness/column that does not fit its public
    /// geometry. Protocol Two masking is not supported by this Protocol One stage driver.
    pub fn prove_stage_windowed_composed<T, R, V>(
        &mut self,
        transcript: &mut T,
        k: u16,
        cut: usize,
        next_cut: usize,
        native: &mut impl NativeProver,
        regenerate: &mut R,
    ) -> Result<super::StageProof, StageError>
    where
        T: tessera_sumcheck::SumcheckTranscript<F> + ?Sized,
        R: FnMut() -> Result<V, StageError>,
        V: WitnessSource,
    {
        if k == 0 || self.parts.len() != usize::from(k) {
            return internal("windowed stages follow the previous stage's splits");
        }
        let proof = if cut == 0 {
            if self.pending_window.is_some() {
                return internal("a prepared window agrees with the stage memory plan");
            }
            let started = Instant::now();
            let proof = self.prove_stage_body(transcript, k, native)?;
            tessera_codec::metrics::pass(tessera_codec::metrics::Pass::P3, started.elapsed());
            proof
        } else {
            let prepared = match self.pending_window.take() {
                Some(prepared) if prepared.index == k && prepared.cut == cut => prepared,
                Some(_) => return internal("prepared windows are consumed in stage order"),
                None => {
                    self.begin_stage(transcript, k, native)?;
                    self.prepare_window(k, cut, native, regenerate)?
                }
            };
            let PreparedStage {
                index: _,
                cut: _,
                arena,
                plans,
                points,
                graph_indices,
                factors,
                prepared,
            } = prepared;
            let mut source = Source {
                arena: &arena,
                factors: &factors,
                native,
                regenerate,
                witness: None,
                started: None,
            };
            let proven = prepared
                .prove(
                    transcript,
                    tessera_sumcheck::Prefix::Stage(u32::from(k)),
                    &mut source,
                )
                .map_err(map_error)?;
            for (round, &r) in (0u32..).zip(proven.point()) {
                self.coords.insert(
                    tessera_compiler::schedule::Coord::Round { stage: k, round },
                    r,
                );
            }
            let stage = &self.schedule.stages[usize::from(k)];
            let mut values = vec![Vec::new(); stage.claims.len()];
            for (i, instance) in stage.instances.iter().enumerate() {
                let at = proven.instance_point(i).ok_or(StageError::Internal {
                    rule: "the window proves every compiled instance",
                })?;
                let evals = proven.instance_evals(i).ok_or(StageError::Internal {
                    rule: "the window sends every compiled evaluation",
                })?;
                let (claims, spread) = if let Relation::Native { tag, .. } = instance.relation {
                    (native.claims(tag, at, evals)?, Vec::new())
                } else {
                    let graph = graph_indices[i].ok_or(StageError::Internal {
                        rule: "a graph instance has its column recipe",
                    })?;
                    let points = plans[graph].claim_points(&arena, &points[graph], at)?;
                    if points.len() != evals.len() {
                        return internal("a regenerated relation sends one value per claim");
                    }
                    (
                        points
                            .into_iter()
                            .zip(evals)
                            .map(|(point, &value)| tessera_sumcheck::Claim { point, value })
                            .collect(),
                        plans[graph].spread(),
                    )
                };
                if claims.len() != instance.claims.len() {
                    return internal("a regenerated relation's claims fit its schedule");
                }
                for (j, (c, claim)) in instance.claims.clone().zip(claims).enumerate() {
                    let spec = &stage.claims[c];
                    let mut expected = self.point(&spec.point)?;
                    if let Layers::Eq(rho) = &spec.layers {
                        expected.extend(self.point(rho)?);
                    }
                    if !self.lands(&claim.point, &expected, spread.contains(&j)) {
                        return internal("a regenerated relation lands at the schedule's point");
                    }
                    values[c] = vec![claim.value];
                }
            }
            self.claims.push(values);
            proven.into_proof()
        };
        if next_cut != 0 {
            let next = k.checked_add(1).ok_or(StageError::Memory)?;
            self.pending_window =
                Some(self.prepare_window(next, next_cut, &super::NoNative, regenerate)?);
        }
        self.finish_stage(transcript, k, proof)
    }
}

impl<W: WitnessSource + ?Sized> StageProver<'_, W> {
    /// Graph-only form of [`Self::prove_stage_windowed_composed`], with the same cuts and
    /// regenerated witness contract.
    ///
    /// # Errors
    /// As [`Self::prove_stage_windowed_composed`]; native descriptors require its composed form.
    pub fn prove_stage_windowed<T, R, V>(
        &mut self,
        transcript: &mut T,
        k: u16,
        cut: usize,
        next_cut: usize,
        regenerate: &mut R,
    ) -> Result<super::StageProof, StageError>
    where
        T: tessera_sumcheck::SumcheckTranscript<F> + ?Sized,
        R: FnMut() -> Result<V, StageError>,
        V: WitnessSource,
    {
        self.prove_stage_windowed_composed(
            transcript,
            k,
            cut,
            next_cut,
            &mut super::NoNative,
            regenerate,
        )
    }
}

impl Plan {
    fn heap_bytes(&self) -> Result<u64, StageError> {
        match self {
            Self::Elementwise { inputs, terms, .. } => terms.iter().try_fold(
                add(
                    array::<Read>(inputs.capacity())?,
                    array::<Term<F>>(terms.capacity())?,
                )?,
                |n, t| {
                    add(
                        n,
                        super::source_memory::growing::<usize>(t.factors().len())?,
                    )
                },
            ),
            Self::Selection { members, .. } => members.iter().try_fold(
                array::<(F, Vec<usize>)>(members.capacity())?,
                |n, (_, indices)| add(n, array::<usize>(indices.capacity())?),
            ),
            _ => Ok(0),
        }
    }

    fn memory_shape(&self, vars: usize) -> tessera_sumcheck::WindowShape {
        let (tables, delta, windowable) = match self {
            Self::Elementwise { inputs, terms, .. } => (
                inputs.len(),
                terms.iter().map(|t| t.factors().len()).max().unwrap_or(1),
                true,
            ),
            Self::MatMulWeight { .. } | Self::MatMulAct { .. } => (2, 2, false),
            Self::Rotate { .. } | Self::Selection { .. } => (2, 2, true),
            Self::Recomposition { h, .. } => (5 + 3 * usize::from(h.is_some()), 3, true),
        };
        tessera_sumcheck::WindowShape {
            vars,
            tables,
            delta,
            windowable,
        }
    }
}

impl<W: WitnessSource + ?Sized> StageProver<'_, W> {
    /// Canonical source metadata and original resident column requests for a
    /// compiled stage, without loading witness data or drawing challenges.
    /// `native_tables` supplies every canonical native operand before sharing.
    /// Public geometry construction allocates; its temporaries are additional.
    /// Native source payloads/instance callbacks, relation contractions, sumcheck
    /// engines, pending windows, messages and runtime/allocator work are separate.
    ///
    /// # Errors
    /// The public stage/operand geometry is invalid or its byte counts overflow.
    pub fn relation_source_memory(
        &self,
        k: u16,
        native_tables: impl Fn(u32) -> Result<usize, StageError>,
    ) -> Result<super::StageSourceMemory, StageError> {
        use super::source_memory::growing;
        let stage = self
            .schedule
            .stages
            .get(usize::from(k))
            .ok_or(StageError::Internal {
                rule: "source memory names a compiled stage",
            })?;
        let mut arena = Arena {
            streaming: true,
            geometry_only: true,
            ..Arena::default()
        };
        let mut factors = Vec::new();
        let (mut graphs, mut native, mut nested, mut points) = (0usize, 0usize, 0u64, 0u64);
        for instance in &stage.instances {
            if let Relation::Native { tag, .. } = instance.relation {
                native = native
                    .checked_add(native_tables(tag)?)
                    .ok_or(StageError::Memory)?;
                continue;
            }
            graphs = graphs.checked_add(1).ok_or(StageError::Memory)?;
            let plan = self.plan(stage, instance, &mut arena)?;
            let layers = match &instance.layers {
                Layers::Eq(rho) => rho.len(),
                _ => 0,
            };
            let len = instance
                .point
                .len()
                .checked_add(layers)
                .ok_or(StageError::Memory)?;
            points = add(points, growing::<F>(len)?)?;
            // Only public lengths control the column/factor recipes. No proof,
            // incoming claim, transcript label or challenge is supplied here.
            let point = vec![F::ZERO; len];
            drop(plan.window(&arena, &point, &mut factors)?);
            nested = add(nested, plan.heap_bytes()?)?;
        }
        let mut metadata = add(nested, points)?;
        metadata = add(
            metadata,
            add(growing::<Plan>(graphs)?, growing::<Vec<F>>(graphs)?)?,
        )?;
        metadata = add(metadata, growing::<Option<usize>>(stage.instances.len())?)?;
        metadata = add(
            metadata,
            growing::<Factor>(
                factors
                    .len()
                    .checked_add(native)
                    .ok_or(StageError::Memory)?,
            )?,
        )?;
        metadata = add(
            metadata,
            add(
                growing::<Column>(arena.secret.len())?,
                growing::<Column>(arena.public.len())?,
            )?,
        )?;
        let mut inputs = 0;
        for column in arena.secret.iter().chain(&arena.public) {
            metadata = add(metadata, column.heap_bytes()?)?;
            inputs = add(inputs, growing::<Goldilocks>(column.len())?)?;
        }
        for factor in &factors {
            metadata = add(metadata, factor.heap_bytes()?)?;
        }
        Ok(super::StageSourceMemory { metadata, inputs })
    }

    /// Additional window specifications and message work, before sharing.
    /// Native terms are bounded by the canonical native instance allocation
    /// reservation; graph terms follow the same Plan variants as `window`.
    pub(crate) fn window_metadata_memory(
        &self,
        k: u16,
        native_shape: impl Fn(u32) -> Result<tessera_sumcheck::WindowShape, StageError>,
        native_terms: u64,
    ) -> Result<tessera_sumcheck::WindowMetadataMemory, StageError> {
        let stage = self
            .schedule
            .stages
            .get(usize::from(k))
            .ok_or(StageError::Internal {
                rule: "window memory names a compiled stage",
            })?;
        let mut arena = Arena {
            streaming: true,
            geometry_only: true,
            ..Arena::default()
        };
        let mut shapes = Vec::new();
        let mut terms = Vec::new();
        for instance in &stage.instances {
            let (shape, count) = match instance.relation {
                Relation::Native { tag, .. } => (native_shape(tag)?, native_terms),
                _ => {
                    let plan = self.plan(stage, instance, &mut arena)?;
                    let count = match &plan {
                        Plan::Elementwise { terms, .. } => terms.len() as u64,
                        Plan::Recomposition { h, .. } => 2 + u64::from(h.is_some()),
                        _ => 1,
                    };
                    (plan.memory_shape(instance.vars as usize), count)
                }
            };
            shapes.push(shape);
            terms.push(count);
        }
        tessera_sumcheck::WindowMetadataMemory::new::<F>(&shapes, &terms, 7).map_err(map_error)
    }

    /// Public operand/grid geometry; called before P1 with a source that lends no witness data.
    pub(crate) fn window_shapes(
        &self,
        k: u16,
        native: impl Fn(u32) -> Result<tessera_sumcheck::WindowShape, StageError>,
    ) -> Result<Vec<tessera_sumcheck::WindowShape>, StageError> {
        let stage = self
            .schedule
            .stages
            .get(usize::from(k))
            .ok_or(StageError::Internal {
                rule: "a stage memory plan names a compiled stage",
            })?;
        let mut arena = Arena {
            streaming: true,
            geometry_only: true,
            ..Arena::default()
        };
        stage
            .instances
            .iter()
            .map(|instance| match instance.relation {
                Relation::Native { tag, .. } => native(tag),
                _ => Ok(self
                    .plan(stage, instance, &mut arena)?
                    .memory_shape(instance.vars as usize)),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::relations::build;
    use super::*;
    use tessera_codec::{Ascii, SecretBuf};
    use tessera_field::QuadraticExtension;
    use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
    use tessera_ir::TensorId;
    use tessera_sumcheck::{ElementwiseShape, Prefix, RightLayout, prove_batch};

    // Public synthetic source data, including nonzero unused/padded bit slots. The recipes
    // must agree as polynomials for arbitrary field data, not only an honest zero-check.
    #[derive(Clone)]
    struct Synthetic(Vec<Vec<i64>>);
    impl WitnessSource for Synthetic {
        fn values(&self, tensor: TensorId) -> Option<&[i64]> {
            self.0.get(tensor.0 as usize).map(Vec::as_slice)
        }
    }
    fn point(n: usize, seed: u64) -> Vec<F> {
        (0..n)
            .map(|j| {
                if seed < 2 {
                    F::from(Goldilocks::from_u64_reduced(seed))
                } else {
                    F::new(
                        Goldilocks::from_u64_reduced(j as u64 + seed),
                        Goldilocks::from_u64_reduced(2 * j as u64 + 1),
                    )
                }
            })
            .collect()
    }
    fn transcript() -> Transcript {
        Transcript::new(&SessionContext {
            protocol: Ascii::new("tessera-protocol/1").unwrap(),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new("test-small/1").unwrap()),
            statement_digest: [45; 32],
            session_id: [59; 16],
        })
    }
    fn compare(plans: &[Plan], points: &[Vec<F>], secret: &[usize], public: &[usize]) {
        let witness = Synthetic(
            secret
                .iter()
                .enumerate()
                .map(|(k, &len)| {
                    (0..len)
                        .map(|i| ((3 * i + 7 * k) % 101) as i64 - 50)
                        .collect()
                })
                .collect(),
        );
        let mut literal = Arena::default();
        let mut recipe = Arena {
            streaming: true,
            ..Arena::default()
        };
        for (i, values) in witness.0.iter().enumerate() {
            literal.secret.push(Column::Secret(
                values
                    .iter()
                    .map(|&v| Goldilocks::from_i64(v))
                    .collect::<SecretBuf<_>>(),
            ));
            recipe.secret.push(Column::Tensor {
                tensors: vec![TensorId(i as u32)],
                shape: vec![values.len() as u32],
                block: values.len(),
                len: values.len(),
                padded: true,
            });
        }
        for (k, &len) in public.iter().enumerate() {
            let values: Vec<_> = (0..len)
                .map(|i| Goldilocks::from_u64_reduced((i * i + 3 * k + 1) as u64))
                .collect();
            literal.public.push(Column::Public(values.clone()));
            recipe.public.push(
                Column::generated(
                    true,
                    len,
                    array::<Goldilocks>(values.capacity()).unwrap(),
                    move |i| Ok(values[i]),
                )
                .unwrap(),
            );
        }
        let (built, degree) = build(&literal, plans, points).unwrap();
        let instances: Vec<_> = built.iter().map(|b| b.instance()).collect();
        let mut reference_transcript = transcript();
        let reference = prove_batch(
            &mut reference_transcript,
            Protocol::One,
            Prefix::Stage(3),
            degree,
            &instances,
        )
        .unwrap();
        let reference_after = reference_transcript.challenge_fp2("after");
        let mut factors = Vec::new();
        let specs: Vec<_> = plans
            .iter()
            .zip(points)
            .map(|(p, r)| p.window(&recipe, r, &mut factors).unwrap())
            .collect();
        for (plan, spec) in plans.iter().zip(&specs) {
            let charged = plan.memory_shape(spec.vars);
            assert_eq!(charged.tables, spec.tables.len());
            assert_eq!(
                charged.delta,
                spec.terms.iter().map(|t| t.factors().len()).max().unwrap()
            );
            assert_eq!(charged.windowable, spec.windowed);
        }
        let rounds = specs.iter().map(|s| s.vars).max().unwrap();
        for cut in 0..rounds.max(1) {
            let mut regenerate = || Ok(witness.clone());
            let mut source = Source {
                arena: &recipe,
                factors: &factors,
                native: &super::super::NoNative,
                regenerate: &mut regenerate,
                witness: None,
                started: None,
            };
            let prepared =
                PreparedWindow::new(&mut source, Protocol::One, degree, specs.clone(), cut)
                    .unwrap();
            let mut actual_transcript = transcript();
            let actual = prepared
                .prove(&mut actual_transcript, Prefix::Stage(3), &mut source)
                .unwrap();
            assert_eq!(actual.proof(), reference.proof(), "cut {cut}");
            assert_eq!(actual_transcript.challenge_fp2("after"), reference_after);
            for (i, plan) in plans.iter().enumerate() {
                let at = actual.instance_point(i).unwrap();
                let evals = actual.instance_evals(i).unwrap();
                let expected = built[i].claims(at, evals).unwrap();
                let mapped = plan.claim_points(&recipe, &points[i], at).unwrap();
                assert_eq!(
                    mapped,
                    expected.into_iter().map(|c| c.point).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn regenerated_elementwise_and_broadcast_columns_match_literal_relations() {
        for seed in 0..=2 {
            compare(
                &[
                    Plan::Elementwise {
                        inputs: vec![
                            Read::Operand(0),
                            Read::Broadcast {
                                table: 1,
                                low_vars: 1,
                            },
                            Read::Public(0),
                            Read::Spread(2),
                        ],
                        terms: vec![
                            Term::new(F::ONE, vec![0, 1]),
                            Term::new(-F::ONE, vec![2, 3]),
                            Term::new(F::ONE, vec![0, 0]),
                        ],
                        shape: ElementwiseShape::elementwise(5),
                        point_from: 0,
                    },
                    Plan::Elementwise {
                        inputs: vec![Read::Operand(0)],
                        terms: vec![Term::new(F::ONE, vec![0, 0])],
                        shape: ElementwiseShape {
                            vars: 5,
                            summed_vars: 2,
                        },
                        point_from: 2,
                    },
                ],
                &[point(5, seed), point(5, seed)],
                &[32, 4, 32],
                &[32],
            );
        }
    }
    #[test]
    fn regenerated_rope_and_combined_selection_kernels_match_literal_relations() {
        for seed in 0..=2 {
            compare(
                &[
                    Plan::Rotate {
                        x: 0,
                        cos: 0,
                        sin: 1,
                        bit: 1,
                        vars: 5,
                    },
                    Plan::Selection {
                        x: 0,
                        low: 1,
                        axis: 3,
                        high: 1,
                        rows: 2,
                        members: vec![(F::ONE, vec![1, 4, 7]), (-F::ONE, vec![7, 3, 1])],
                    },
                ],
                &[point(5, seed), point(4, seed)],
                &[32],
                &[16, 16],
            );
        }
    }
    #[test]
    fn regenerated_recomposition_weights_extensions_and_claims_match_literal_relations() {
        for (low, high) in [(1u8, 0u8), (3, 2), (5, 9), (9, 1)] {
            for rows in [0, 2] {
                for public in [false, true] {
                    let kb = u32::from(low).next_power_of_two().trailing_zeros() as usize;
                    let kh = u32::from(high).next_power_of_two().trailing_zeros() as usize;
                    let k = kb.max(kh);
                    let plans = [Plan::Recomposition {
                        x: if public {
                            Read::Public(0)
                        } else {
                            Read::Operand(0)
                        },
                        b: 1,
                        h: (high != 0).then_some(2),
                        low,
                        high,
                        rows,
                        point_from: k,
                    }];
                    compare(
                        &plans,
                        &[point(rows + k, 7)],
                        &[1 << rows, 1 << (rows + kb), 1 << (rows + kh)],
                        &[1 << rows],
                    );
                }
            }
        }
    }
    #[test]
    fn regenerated_matrix_contractions_stay_resident_and_match_both_layouts() {
        use tessera_sumcheck::{MatMulActShape, MatMulWeightShape};
        let mw = MatMulWeightShape {
            layer_vars: 1,
            token_vars: 1,
            in_vars: 3,
            out_vars: 1,
        };
        let ma = MatMulActShape {
            stack_vars: 1,
            row_vars: 1,
            inner_vars: 3,
            col_vars: 1,
        };
        compare(
            &[
                Plan::MatMulWeight {
                    w: 0,
                    x: 1,
                    shape: mw,
                },
                Plan::MatMulAct {
                    a: 0,
                    b: 1,
                    layout: RightLayout::Direct,
                    shape: ma,
                },
                Plan::MatMulAct {
                    a: 0,
                    b: 1,
                    layout: RightLayout::Transposed,
                    shape: ma,
                },
                Plan::Elementwise {
                    inputs: vec![Read::Operand(0)],
                    terms: vec![Term::new(F::ONE, vec![0, 0])],
                    shape: ElementwiseShape::elementwise(5),
                    point_from: 0,
                },
            ],
            &[point(3, 3), point(3, 3), point(3, 3), point(5, 3)],
            &[32, 32],
            &[],
        );
    }
    #[test]
    fn factor_heap_counts_owned_points_and_selection_rows() {
        for vars in [0usize, 1, 17, 62] {
            let mut held = None;
            let measured = allocation_counter::measure(|| {
                held = Some(eq_factor(vec![F::ZERO; vars], 0, 0).unwrap());
            });
            assert!(
                u64::try_from(measured.bytes_current).unwrap()
                    <= held.as_ref().unwrap().heap_bytes().unwrap()
            );
        }
        let mut held = None;
        let measured = allocation_counter::measure(|| {
            let plan = Plan::Selection {
                x: 0,
                low: 1,
                axis: 3,
                high: 1,
                rows: 10,
                members: vec![(F::ONE, vec![0; 1024]); 32],
            };
            let arena = Arena::default();
            let mut factors = Vec::new();
            drop(plan.window(&arena, &[F::ZERO; 12], &mut factors).unwrap());
            held = Some(factors);
        });
        let factors = held.as_ref().unwrap();
        let actual = factors
            .iter()
            .try_fold(array::<Factor>(factors.capacity()).unwrap(), |n, f| {
                add(n, f.heap_bytes().unwrap())
            })
            .unwrap();
        assert!(u64::try_from(measured.bytes_current).unwrap() <= actual);
    }
}
