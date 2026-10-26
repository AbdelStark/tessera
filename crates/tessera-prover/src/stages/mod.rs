// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reduction stages of a stage schedule (RFC-0010 'Stages', RFC-0037), proven over a
//! [`WitnessSource`].
//!
//! A [`StageProver`] walks a [`StageSchedule`] in order:
//!
//! 1. **The leaf stage** leaves its claims at its points. [`StageProver::leaf`] computes the
//!    graph claims there from the witness. [`StageProver::leaf_split_values`] supplies their
//!    atom splits to [`crate::PreparedLeaf::finish`] for one canonical `stage/0/split` message;
//!    [`StageProver::leaf_splits`] also supports standalone graph proofs.
//! 2. **Stage `k`**, [`StageProver::prove_stage`], proves one batched sumcheck.
//!    - In stage 1 it first squeezes the layer point `ρ_1` (`stage/1/layer/{c}`), which stacks
//!      the leaf stage's per-layer values, and the zero-check point (`stage/1/zero/{c}`).
//!    - Each instance is its relation over the witness's tables at its claim's point:
//!      `MatMulWeight`, `MatMulAct`, `Elementwise` for products, sums of squares and the equality
//!      zero-checks, and `Rotate`'s kernel. Stacked instances stack their kinds' layers.
//!    - A relation's public operand is read as `Public` and leaves no claim (RFC-0072 'Public
//!      tensors' 'As an operand'): a `Mul`'s by a public tensor, the constants of a `ConstMul` by
//!      one constant per channel and of a committed `ConstAdd`'s zero-check.
//!    - After the evaluations, the stage sends its split claims' values under `stage/{k}/split`,
//!      claim by claim and part by part (RFC-0037 'Splits'). A claim's public terms, which
//!      [`public_value`] evaluates, move to its value first: a ready claim's instance or pool
//!      claim is its value less them, and a claim of public terms alone has no part.
//!    - [`StageProver::prove_stage_composed`] interleaves native tables in that same batch and
//!      routes their transformed original graph sources through those same splits and pool.
//! 3. **The pool** ends with every committed part's value, [`StageProver::pool`]: what the
//!    openings prove (RFC-0011, RFC-0012).
//!
//! Every claim an instance leaves is checked against the schedule's point for it, so a relation
//! and the schedule cannot disagree silently. The witness's tables are secret-class (05 §3): this
//! module borrows them, and keeps what it derives in `SecretBuf`s that clear their memory.

mod public;
mod relations;
mod tables;

use std::collections::HashMap;
use std::fmt;

use tessera_codec::error::ErrorCode;
use tessera_compiler::CompileError;
use tessera_compiler::Lowering;
use tessera_compiler::schedule::{
    Atom, Claim, Coord, Factor, Layers, NodeRef, Part, Relation, Source, StageSchedule,
    nonzero_at_padding,
};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_ir::{Graph, Op, TensorId};
use tessera_poly::{eq, evaluate_base, less_than};
use tessera_sumcheck::{
    Prefix, SumcheckError, SumcheckProof, SumcheckTranscript, prove_batch, zero_check_point,
};
use tessera_tables::TableSpec;

use relations::{Arena, build};

pub use public::{PublicTensors, public_value};

/// Public points drawn once for a complete mixed stage, before its gamma.
pub struct NativeContext<'a> {
    /// Stage index.
    pub stage: u16,
    /// Earlier layer point (empty for no layer coordinates).
    pub layer: &'a [GoldilocksExt2],
    /// Complete stage zero-check point.
    pub zero: &'a [GoldilocksExt2],
}

/// Native CPU tables and their ORIGINAL graph-source transformations. Implementations retain
/// derived witness arrays in clearing buffers and lend opaque sumcheck tables to the engine.
/// Fixed incoming targets must come from the earlier canonical leaf bindings before gamma.
pub trait NativeProver {
    /// Prepare this stage's tables at the already drawn public points, BEFORE gamma.
    fn prepare(&mut self, context: NativeContext<'_>) -> Result<(), StageError>;
    /// Fixed incoming native target; zero-checks must return zero.
    fn claim(&self, tag: u32) -> Result<GoldilocksExt2, StageError>;
    /// Native raw tables/terms; public equality/selectors are marked public as usual.
    fn instance(
        &self,
        tag: u32,
    ) -> Result<tessera_sumcheck::Instance<'_, GoldilocksExt2>, StageError>;
    /// Transform the SAME raw final evaluations into graph claims in descriptor order. Points
    /// retain layer coordinates; every other source/auxiliary/opening debt is still owed.
    fn claims(
        &self,
        tag: u32,
        point: &[GoldilocksExt2],
        values: &[GoldilocksExt2],
    ) -> Result<Vec<tessera_sumcheck::Claim<GoldilocksExt2>>, StageError>;
}
struct NoNative;
impl NativeProver for NoNative {
    fn prepare(&mut self, _: NativeContext<'_>) -> Result<(), StageError> {
        Ok(())
    }
    fn claim(&self, _: u32) -> Result<GoldilocksExt2, StageError> {
        internal("a composed stage requires its native backend")
    }
    fn instance(
        &self,
        _: u32,
    ) -> Result<tessera_sumcheck::Instance<'_, GoldilocksExt2>, StageError> {
        internal("a composed stage requires its native backend")
    }
    fn claims(
        &self,
        _: u32,
        _: &[GoldilocksExt2],
        _: &[GoldilocksExt2],
    ) -> Result<Vec<tessera_sumcheck::Claim<GoldilocksExt2>>, StageError> {
        internal("a composed stage requires its native backend")
    }
}

/// The witness a stage proves over: every tensor's values, true integers, flat and row-major.
/// They are secret-class (05 §3): a source lends them, by reference and for a call.
pub trait WitnessSource {
    /// The values of `tensor`, or `None` when the witness lacks it.
    fn values(&self, tensor: TensorId) -> Option<&[i64]>;
}

/// The CPU engine's witness, every tensor in memory.
impl WitnessSource for tessera_witness::Witness {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        self.tensor(tensor).map(|buffer| &buffer[..])
    }
}

/// A reduction stage's messages (RFC-0036 'StageProof'): its sumcheck, then the values of the
/// claims it splits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageProof {
    /// The stage, from 1.
    pub stage: u16,
    /// Its rounds and final evaluations (RFC-0010 'Sumcheck messages').
    pub sumcheck: SumcheckProof<GoldilocksExt2>,
    /// The values of its split claims' parts, claim by claim (RFC-0037 'Splits').
    pub splits: Vec<GoldilocksExt2>,
}

/// Why a stage cannot be proven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageError {
    /// The schedule names what the stages do not prove, or disagrees with a relation or the
    /// witness: a bug (`ERR-INTERNAL`).
    Internal {
        /// The rule broken.
        rule: &'static str,
    },
    /// A relation refused its tables (`ERR-INTERNAL`).
    Sumcheck(SumcheckError),
    /// A table does not fit in memory (`ERR-RES-MEMORY`).
    Memory,
}

impl StageError {
    /// The stable code of 04 §3.
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Internal { .. } | Self::Sumcheck(_) => ErrorCode::Internal,
            Self::Memory => ErrorCode::ResMemory,
        }
    }
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal { rule } => write!(f, "the stages break a rule: {rule}"),
            Self::Sumcheck(error) => write!(f, "a relation refused its tables: {error}"),
            Self::Memory => f.write_str("a stage's table does not fit in memory"),
        }
    }
}

impl std::error::Error for StageError {}

impl From<SumcheckError> for StageError {
    fn from(error: SumcheckError) -> Self {
        Self::Sumcheck(error)
    }
}

/// Proves a [`StageSchedule`]'s stages over a [`WitnessSource`], in order.
pub struct StageProver<'a, W: WitnessSource + ?Sized> {
    /// The graph the schedule was derived from.
    graph: &'a Graph,
    /// The tables its nodes name, RoPE's among them.
    tables: &'a [TableSpec],
    schedule: &'a StageSchedule,
    witness: &'a W,
    /// The graph's public tensors (RFC-0072 'Public tensors').
    public: PublicTensors,
    /// The challenges drawn so far, by coordinate.
    coords: HashMap<Coord, GoldilocksExt2>,
    /// Each stage's claims' values: one per layer for a leaf claim in the layers, else one.
    claims: Vec<Vec<Vec<GoldilocksExt2>>>,
    /// Each stage's parts' values, as its claims'.
    parts: Vec<Vec<Vec<Vec<GoldilocksExt2>>>>,
    /// The pool's values, as the stages reach them.
    pool: Vec<Option<GoldilocksExt2>>,
}

/// Shows the progress, never a value (05 §3).
impl<W: WitnessSource + ?Sized> fmt::Debug for StageProver<'_, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageProver")
            .field("stages", &self.claims.len())
            .finish_non_exhaustive()
    }
}

impl<'a, W: WitnessSource + ?Sized> StageProver<'a, W> {
    /// A prover of `schedule`, derived from `lowering`, over `witness`.
    pub fn new(lowering: &'a Lowering, schedule: &'a StageSchedule, witness: &'a W) -> Self {
        Self::for_graph(&lowering.graph, &lowering.tables, schedule, witness)
    }

    /// A prover of `schedule`, derived from `graph` and its tables `tables` with no lowering, over
    /// `witness`: a graph of the anchor's circuits, with no layers and no logits
    /// ([`derive_graph_schedule`](tessera_compiler::derive_graph_schedule), RFC-0072 'Graphs
    /// without layers'). It proves what [`new`](Self::new) proves for a lowering of that graph.
    ///
    /// **Precondition.** As for the schedule, `graph` has passed RFC-0006 range analysis
    /// ([`analyze`](tessera_compiler::analyze)), and `witness` holds every tensor of it. Its tables
    /// pad every tensor with zeros, so [`leaf`](Self::leaf) refuses a graph that holds a lookup
    /// non-zero at padding with padded positions (RFC-0072 'Padded positions').
    pub fn for_graph(
        graph: &'a Graph,
        tables: &'a [TableSpec],
        schedule: &'a StageSchedule,
        witness: &'a W,
    ) -> Self {
        Self {
            graph,
            tables,
            schedule,
            witness,
            public: PublicTensors::of(graph),
            coords: HashMap::new(),
            claims: Vec::new(),
            parts: Vec::new(),
            pool: vec![None; schedule.pool.len()],
        }
    }

    /// The value of `coord`, once drawn: `½` for a sum's.
    pub fn challenge(&self, coord: Coord) -> Option<GoldilocksExt2> {
        if coord == Coord::Half {
            return GoldilocksExt2::from(Goldilocks::from_u64_reduced(2)).inverse();
        }
        self.coords.get(&coord).copied()
    }

    /// The leaf stage's claims at `points`, site `s`'s at `points[s]` (#118's leaf layout), with
    /// their true values from the witness: one per layer for a site in the layers, else one.
    ///
    /// # Errors
    ///
    /// [`StageError::Internal`] when the graph holds a lookup non-zero at padding with padded
    /// positions, a point has the wrong number of coordinates or the leaf stage was proven already;
    /// [`StageError::Memory`] when a table does not fit.
    pub fn leaf(
        &mut self,
        points: &[Vec<GoldilocksExt2>],
    ) -> Result<Vec<Vec<GoldilocksExt2>>, StageError> {
        self.check_padding()?;
        let leaf = &self.schedule.stages[0];
        if !self.claims.is_empty() || points.len() != leaf.claims.len() {
            return internal("the leaf stage comes first, with one point per site");
        }
        for (s, (claim, point)) in leaf.claims.iter().zip(points).enumerate() {
            if point.len() != claim.point.len() {
                return internal("a leaf point has its site's coordinates");
            }
            let site = u32::try_from(s).unwrap_or(u32::MAX);
            for (index, &value) in (0u32..).zip(point) {
                self.coords.insert(Coord::Leaf { site, index }, value);
            }
        }
        let mut values = Vec::with_capacity(leaf.claims.len());
        for claim in &leaf.claims {
            let mut per_layer = Vec::new();
            for layer in self.layers_of(claim) {
                per_layer.push(self.claim_value(claim, layer)?);
            }
            values.push(per_layer);
        }
        self.claims.push(values.clone());
        Ok(values)
    }

    /// Refuses a graph holding a lookup non-zero at padding whose output has padded positions
    /// (RFC-0072 'Padded positions'): they hold its table's value at the zero key, such as
    /// `AesSbox(0) = 0x63`, where the stage tables, padded with zeros, hold 0. #526's fourth slice
    /// computes the tables over padded shapes.
    fn check_padding(&self) -> Result<(), StageError> {
        for node in &self.graph.nodes {
            let padded = node.outputs.iter().any(|t| {
                self.graph
                    .tensors
                    .get(t.0 as usize)
                    .is_some_and(|spec| spec.shape.iter().any(|len| !len.is_power_of_two()))
            });
            if !padded {
                continue;
            }
            match nonzero_at_padding(self.graph, self.tables, node) {
                Ok(false) => {}
                Ok(true) => {
                    return internal(
                        "a lookup non-zero at padding has no padded positions while the stage \
                         tables pad with zeros",
                    );
                }
                Err(CompileError::Schedule { rule, .. }) => return internal(rule),
                Err(_) => return internal("a lookup's padding is decided from public data"),
            }
        }
        Ok(())
    }

    /// Stage 0's splits (RFC-0037 'Splits'): the values of the leaf claims' parts, claim by claim,
    /// layer by layer and part by part, absorbed under `stage/0/split`.
    ///
    /// # Errors
    ///
    /// [`StageError::Internal`] before [`leaf`](Self::leaf), or after the splits;
    /// [`StageError::Memory`] when a table does not fit.
    pub fn leaf_splits<T: SumcheckTranscript<GoldilocksExt2> + ?Sized>(
        &mut self,
        transcript: &mut T,
    ) -> Result<Vec<GoldilocksExt2>, StageError> {
        let splits = self.leaf_split_values()?;
        transcript.absorb_message("stage/0/split", &splits);
        Ok(splits)
    }

    /// Compute stage 0's graph atom split values without absorbing a transcript message.
    ///
    /// For a leaf stage with lookup tensor values, pass these values to
    /// [`PreparedLeaf::finish`](crate::leaf::PreparedLeaf::finish) so that the two parts are
    /// absorbed as one vector (RFC-0037/0045). The returned values follow claim, real-layer
    /// and part order. This advances the graph prover exactly as [`leaf_splits`](Self::leaf_splits)
    /// does; call it only once, after [`leaf`](Self::leaf).
    ///
    /// # Errors
    /// [`StageError::Internal`] before the leaf claims or after their splits;
    /// [`StageError::Memory`] when a table does not fit.
    pub fn leaf_split_values(&mut self) -> Result<Vec<GoldilocksExt2>, StageError> {
        if self.claims.len() != 1 || !self.parts.is_empty() {
            return internal("the leaf stage's splits follow its claims");
        }
        let splits = self.split_values(0)?;
        // With no reduction stage, as for a graph whose leaf claims all reach committed or
        // public tensors, no stage 1 draws `ρ_1` after the leaf stage: its pool values are final
        // here (RFC-0072 'No reduction stage'). The schedule refuses a graph with layers and no
        // stage 1, so none of them is stacked.
        if self.schedule.stages.len() == 1 {
            self.fill_pool(0)?;
        }
        Ok(splits)
    }

    /// Proves stage `k`, from 1: its points in stage 1, its batched sumcheck, then its splits.
    ///
    /// # Errors
    ///
    /// [`StageError::Internal`] out of order, or when a relation's claim is not the schedule's;
    /// [`StageError::Sumcheck`] when a relation refuses its tables; [`StageError::Memory`] when a
    /// table does not fit.
    pub fn prove_stage<T: SumcheckTranscript<GoldilocksExt2> + ?Sized>(
        &mut self,
        transcript: &mut T,
        k: u16,
    ) -> Result<StageProof, StageError> {
        self.prove_stage_composed(transcript, k, &mut NoNative)
    }

    /// Prove ONE mixed native/graph batch and ONE split message. Native tables are prepared
    /// before gamma; raw native evaluations are transformed to the schedule's original graph
    /// claims, which follow the same split/pool routing as graph evaluations.
    ///
    /// # Errors
    /// Existing stage errors, plus `Internal` for inconsistent native shapes/counts/targets
    /// or source points. Native reserves and sumcheck failures retain their error types.
    pub fn prove_stage_composed<T: SumcheckTranscript<GoldilocksExt2> + ?Sized>(
        &mut self,
        transcript: &mut T,
        k: u16,
        native: &mut impl NativeProver,
    ) -> Result<StageProof, StageError> {
        let index = usize::from(k);
        if index == 0 || self.parts.len() != index {
            return internal("the stages are proven in order, after the leaf stage's splits");
        }
        let stage = self
            .schedule
            .stages
            .get(index)
            .ok_or(StageError::Internal {
                rule: "the schedule has the stage",
            })?;
        if k == 1 {
            for c in 0..self.schedule.layers.vars {
                let rho = transcript.challenge(&format!("stage/1/layer/{c}"));
                self.coords.insert(Coord::Layer(c), rho);
            }
            if stage.zero_vars > 0 {
                let z = zero_check_point(transcript, 1, stage.zero_vars as usize);
                for (c, value) in (0u32..).zip(z) {
                    self.coords.insert(Coord::Zero(c), value);
                }
            }
            self.fill_pool(0)?;
        }
        let layer: Vec<_> = (0..self.schedule.layers.vars)
            .map(|c| {
                self.challenge(Coord::Layer(c)).ok_or(StageError::Internal {
                    rule: "the stage has its earlier layer point",
                })
            })
            .collect::<Result<_, _>>()?;
        let zero: Vec<_> = (0..stage.zero_vars)
            .map(|c| {
                self.challenge(Coord::Zero(c)).ok_or(StageError::Internal {
                    rule: "the mixed stage has its zero point",
                })
            })
            .collect::<Result<_, _>>()?;
        if stage
            .instances
            .iter()
            .any(|i| matches!(i.relation, Relation::Native { .. }))
        {
            native.prepare(NativeContext {
                stage: k,
                layer: &layer,
                zero: &zero,
            })?;
        }
        let mut arena = Arena::default();
        let mut plans = Vec::new();
        let mut points = Vec::new();
        let mut graph_indices = Vec::with_capacity(stage.instances.len());
        for instance in &stage.instances {
            if matches!(instance.relation, Relation::Native { .. }) {
                graph_indices.push(None);
                continue;
            }
            graph_indices.push(Some(plans.len()));
            plans.push(self.plan(stage, instance, &mut arena)?);
            let mut point = self.point(&instance.point)?;
            if let Layers::Eq(rho) = &instance.layers {
                point.extend(self.point(rho)?);
            }
            points.push(point);
        }
        let (built, mut degree) = build(&arena, &plans, &points)?;
        let mut instances = Vec::with_capacity(stage.instances.len());
        for (i, instance) in stage.instances.iter().enumerate() {
            if let Relation::Native {
                tag,
                degree: own_degree,
                evaluations,
                zero,
            } = instance.relation
            {
                let claim = native.claim(tag)?;
                if zero && claim != GoldilocksExt2::ZERO {
                    return internal("a native zero-check has the fixed target zero");
                }
                let own = native.instance(tag)?;
                let width = 1usize
                    .checked_shl(instance.vars)
                    .ok_or(StageError::Memory)?;
                if own.tables().iter().any(|t| t.len() != width)
                    || own.tables().iter().filter(|t| !t.is_public()).count() != evaluations
                {
                    return internal("native raw tables fit the compiled instance");
                }
                degree = degree.max(own_degree);
                instances.push(own);
            } else {
                instances.push(
                    built[graph_indices[i].ok_or(StageError::Internal {
                        rule: "a graph instance has its tables",
                    })?]
                    .instance(),
                );
            }
        }
        let proven = prove_batch(transcript, Prefix::Stage(u32::from(k)), degree, &instances)?;
        for (round, &r) in (0u32..).zip(proven.point()) {
            self.coords.insert(Coord::Round { stage: k, round }, r);
        }
        // The claims the evaluations leave, each where the schedule says.
        let mut values = vec![Vec::new(); stage.claims.len()];
        for (i, instance) in stage.instances.iter().enumerate() {
            let point = proven.instance_point(i).ok_or(StageError::Internal {
                rule: "the batch proves every instance",
            })?;
            let evals = proven.instance_evals(i).ok_or(StageError::Internal {
                rule: "the batch proves every instance",
            })?;
            let (claims, spread) = if let Relation::Native { tag, .. } = instance.relation {
                (native.claims(tag, point, evals)?, Vec::new())
            } else {
                let graph = graph_indices[i].ok_or(StageError::Internal {
                    rule: "the instance has graph tables",
                })?;
                (built[graph].claims(point, evals)?, plans[graph].spread())
            };
            if claims.len() != instance.claims.len() {
                return internal("an instance's evaluations are its schedule's claims");
            }
            for (j, (c, claim)) in instance.claims.clone().zip(&claims).enumerate() {
                let spec = &stage.claims[c];
                let mut expected = self.point(&spec.point)?;
                if let Layers::Eq(rho) = &spec.layers {
                    expected.extend(self.point(rho)?);
                }
                if !self.lands(&claim.point, &expected, spread.contains(&j)) {
                    return internal("a relation's claim is at the schedule's point");
                }
                values[c] = vec![claim.value];
            }
        }
        self.claims.push(values);
        let splits = self.split_values(index)?;
        transcript.absorb_message(&format!("stage/{index}/split"), &splits);
        self.fill_pool(index)?;
        Ok(StageProof {
            stage: k,
            sumcheck: proven.into_proof(),
            splits,
        })
    }

    /// The pool's claims' values (RFC-0037 'The claim pool'), each once its stage is proven.
    pub fn pool(&self) -> &[Option<GoldilocksExt2>] {
        &self.pool
    }

    /// Whether a relation's claim at `actual` is the schedule's at `expected`. An operand laid out
    /// over the rows (`spread`) is claimed at the rows' coordinates too, between its channels' and
    /// its layers': the schedule's point has the channels' and the layers' alone.
    fn lands(&self, actual: &[GoldilocksExt2], expected: &[GoldilocksExt2], spread: bool) -> bool {
        if !spread {
            return actual == expected;
        }
        let lambda = self.schedule.layers.vars as usize;
        let Some(low) = expected.len().checked_sub(lambda) else {
            return false;
        };
        actual.len() >= expected.len()
            && actual[..low] == expected[..low]
            && actual[actual.len() - lambda..] == expected[low..]
    }

    /// Records the parts' values of stage `index`'s claims and returns the split ones.
    fn split_values(&mut self, index: usize) -> Result<Vec<GoldilocksExt2>, StageError> {
        let stage = &self.schedule.stages[index];
        let mut splits = Vec::new();
        let mut parts = Vec::with_capacity(stage.claims.len());
        for (c, claim) in stage.claims.iter().enumerate() {
            let layers = self.layers_of(claim);
            let mut values: Vec<Vec<GoldilocksExt2>> =
                vec![Vec::with_capacity(layers.len()); claim.parts.len()];
            // Its public terms move to its value (RFC-0037 'Splits'): a ready claim's one part is
            // the rest. They are outside the layers, so the same in every layer.
            let public = self.public_sum(claim)?;
            for (l, &layer) in layers.iter().enumerate() {
                for (p, part) in claim.parts.iter().enumerate() {
                    let value = if claim.split {
                        let value = self.part_value(claim, part, layer)?;
                        splits.push(value);
                        value
                    } else {
                        self.claims[index][c][l] - public
                    };
                    values[p].push(value);
                }
            }
            parts.push(values);
        }
        // The leaf stage's committed parts are stacked at ρ_1, which stage 1 draws.
        self.parts.push(parts);
        Ok(splits)
    }

    /// The pool's values of stage `index`'s committed parts: a leaf claim's per-layer values
    /// stacked at `ρ_1`.
    fn fill_pool(&mut self, index: usize) -> Result<(), StageError> {
        let stage = &self.schedule.stages[index];
        let rho: Vec<Coord> = (0..self.schedule.layers.vars).map(Coord::Layer).collect();
        for (c, claim) in stage.claims.iter().enumerate() {
            for (p, part) in claim.parts.iter().enumerate() {
                if let Part::Pool { claim: pool, .. } = part {
                    let per_layer = &self.parts[index][c][p];
                    let value = if claim.layers == Layers::Each {
                        self.stack(&rho, per_layer)?
                    } else {
                        per_layer[0]
                    };
                    self.pool[*pool] = Some(value);
                }
            }
        }
        Ok(())
    }

    /// `Σ_ℓ eq(ρ, ℓ)·v_ℓ` over the real layers, the padded ones' values being 0.
    fn stack(
        &self,
        rho: &[Coord],
        values: &[GoldilocksExt2],
    ) -> Result<GoldilocksExt2, StageError> {
        let rho = self.point(rho)?;
        let mut sum = GoldilocksExt2::ZERO;
        for (layer, &value) in (0u64..).zip(values) {
            sum += eq_vertex(&rho, layer) * value;
        }
        Ok(sum)
    }

    /// The value an instance of the next stage proves for part `part` of `claim` of stage
    /// `index`: a leaf claim's per-layer values stacked at `ρ_1`.
    pub(super) fn part_claim(
        &self,
        index: usize,
        claim: usize,
        part: usize,
    ) -> Result<GoldilocksExt2, StageError> {
        let values = &self.parts[index][claim][part];
        if index == 0 && self.schedule.stages[0].claims[claim].layers == Layers::Each {
            let rho: Vec<Coord> = (0..self.schedule.layers.vars).map(Coord::Layer).collect();
            return self.stack(&rho, values);
        }
        Ok(values[0])
    }

    /// The layers a claim's values run over: each real layer for a leaf claim in the layers.
    fn layers_of(&self, claim: &Claim) -> Vec<Option<u32>> {
        match claim.layers {
            Layers::Each => (0..self.schedule.layers.count).map(Some).collect(),
            _ => vec![None],
        }
    }

    /// A claim's true value, in `layer` for a leaf claim in the layers.
    fn claim_value(&self, claim: &Claim, layer: Option<u32>) -> Result<GoldilocksExt2, StageError> {
        let mut value = GoldilocksExt2::ZERO;
        for (l, w) in self.weights(&claim.layers, layer)? {
            value += w * self.extension(claim.kind, l, &claim.point)?;
        }
        Ok(value)
    }

    /// The sum of `claim`'s public terms at their points, each with its coefficient: what the
    /// verifier computes and moves to the claim's value (RFC-0037 'Splits', RFC-0072 'Public
    /// tensors'). The witness holds every tensor, the public ones among them.
    fn public_sum(&self, claim: &Claim) -> Result<GoldilocksExt2, StageError> {
        let mut sum = GoldilocksExt2::ZERO;
        for term in &claim.public {
            let point = self.point(&term.point)?;
            let value = public_value(self.graph, &self.public, term.source, self.witness, &point)?;
            sum += self.coefficient(&term.coefficient, 0)? * value;
        }
        Ok(sum)
    }

    /// A part's value: a virtual atom's without its coefficient, a committed group's with them.
    fn part_value(
        &self,
        claim: &Claim,
        part: &Part,
        layer: Option<u32>,
    ) -> Result<GoldilocksExt2, StageError> {
        match part {
            Part::Instance { atom, .. } => self.atom_value(&claim.atoms[*atom], layer, false),
            Part::Pool { atoms, .. } => {
                let mut sum = GoldilocksExt2::ZERO;
                for &i in atoms {
                    sum += self.atom_value(&claim.atoms[i], layer, true)?;
                }
                Ok(sum)
            }
        }
    }

    /// `Σ_ℓ w(ℓ)·c(ℓ)·X̃_{κ,ℓ}(r)` of an atom, `c` left out when `with_coefficient` is false.
    fn atom_value(
        &self,
        atom: &Atom,
        layer: Option<u32>,
        with_coefficient: bool,
    ) -> Result<GoldilocksExt2, StageError> {
        let mut value = GoldilocksExt2::ZERO;
        for (l, w) in self.weights(&atom.layers, layer)? {
            let c = if with_coefficient {
                self.coefficient(&atom.coefficient, l)?
            } else {
                GoldilocksExt2::ONE
            };
            value += w * c * self.extension(atom.kind, l, &atom.point)?;
        }
        Ok(value)
    }

    /// The layers `layers` weighs, each with its weight: the real layers only, a padded layer's
    /// tensors being 0.
    pub(crate) fn weights(
        &self,
        layers: &Layers,
        each: Option<u32>,
    ) -> Result<Vec<(u32, GoldilocksExt2)>, StageError> {
        let one = GoldilocksExt2::ONE;
        Ok(match layers {
            Layers::None => vec![(0, one)],
            Layers::Each => vec![(
                each.ok_or(StageError::Internal {
                    rule: "a leaf claim's layer",
                })?,
                one,
            )],
            Layers::One(layer) => vec![(*layer, one)],
            Layers::Eq(rho) => {
                let rho = self.point(rho)?;
                (0..self.schedule.layers.count)
                    .map(|l| (l, eq_vertex(&rho, u64::from(l))))
                    .collect()
            }
        })
    }

    /// A coefficient's value in `layer`.
    pub(crate) fn coefficient(
        &self,
        products: &[Vec<Factor>],
        layer: u32,
    ) -> Result<GoldilocksExt2, StageError> {
        let mut sum = GoldilocksExt2::ZERO;
        for product in products {
            let mut value = GoldilocksExt2::ONE;
            for factor in product {
                value *= match factor {
                    Factor::Minus => -GoldilocksExt2::ONE,
                    Factor::Pow2(j) => {
                        GoldilocksExt2::from(Goldilocks::from_u64_reduced(1u64 << j))
                    }
                    Factor::Constant(node) => self.constant(*node, layer)?,
                    Factor::Lt { coords, len } => {
                        let r = self.point(coords)?;
                        less_than(&r, u64::from(*len)).map_err(|_| StageError::Internal {
                            rule: "an Expand's length fits its axis",
                        })?
                    }
                };
            }
            sum += value;
        }
        Ok(sum)
    }

    /// The scalar constant of a `ConstMul`, in `layer` at a template position.
    fn constant(&self, node: NodeRef, layer: u32) -> Result<GoldilocksExt2, StageError> {
        let id = match node {
            NodeRef::Node(id) => id,
            NodeRef::Position(position) => self.schedule.layers.node(position, layer),
        };
        match self.graph.nodes[id.0 as usize].op {
            Op::ConstMul { c } => match self.graph.constants.get(c.0 as usize).map(Vec::as_slice) {
                Some(&[scalar]) => Ok(GoldilocksExt2::from(Goldilocks::from_i64(scalar))),
                _ => internal("a coefficient's ConstMul is by one constant"),
            },
            _ => internal("a coefficient's constant is a ConstMul's"),
        }
    }

    /// `X̃(point)` of `kind`'s tensor in `layer`.
    fn extension(
        &self,
        kind: tessera_compiler::schedule::Kind,
        layer: u32,
        point: &[Coord],
    ) -> Result<GoldilocksExt2, StageError> {
        let table = self.slice(kind, layer)?;
        let r = self.point(point)?;
        evaluate_base::<GoldilocksExt2>(&table, &r).map_err(|_| StageError::Internal {
            rule: "a claim's point has its tensor's coordinates",
        })
    }

    /// The values of `coords`, every one drawn.
    pub(crate) fn point(&self, coords: &[Coord]) -> Result<Vec<GoldilocksExt2>, StageError> {
        coords
            .iter()
            .map(|&c| {
                self.challenge(c).ok_or(StageError::Internal {
                    rule: "a claim's coordinates are drawn before it",
                })
            })
            .collect()
    }

    /// Every instance claim of stage `k`, in instance order: a virtual part's value, or 0 for a
    /// zero-check.
    pub fn instance_claims(&self, k: u16) -> Result<Vec<GoldilocksExt2>, StageError> {
        let stage = self
            .schedule
            .stages
            .get(usize::from(k))
            .ok_or(StageError::Internal {
                rule: "the schedule has the stage",
            })?;
        stage
            .instances
            .iter()
            .map(|instance| match instance.source {
                Source::Native(_) => internal("native targets belong to the composed backend"),
                Source::Part { claim, part } => self.part_claim(usize::from(k) - 1, claim, part),
                Source::ZeroCheck => Ok(GoldilocksExt2::ZERO),
            })
            .collect()
    }
}

/// `eq(ρ, ℓ)` at the vertex `ℓ`, `ρ[0]` its lowest bit.
fn eq_vertex(rho: &[GoldilocksExt2], layer: u64) -> GoldilocksExt2 {
    let bits: Vec<GoldilocksExt2> = (0..rho.len())
        .map(|j| {
            if (layer >> j) & 1 == 1 {
                GoldilocksExt2::ONE
            } else {
                GoldilocksExt2::ZERO
            }
        })
        .collect();
    eq(rho, &bits).unwrap_or(GoldilocksExt2::ZERO)
}

fn internal<T>(rule: &'static str) -> Result<T, StageError> {
    Err(StageError::Internal { rule })
}
