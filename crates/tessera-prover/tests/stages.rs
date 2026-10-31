// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reduction stages of the conformance models (RFC-0010 'Stages', RFC-0037), proven over the
//! CPU engine's witness and checked by a verifier written here from RFC-0010 'Sumcheck messages'
//! and 'Batching within a stage' and RFC-0037 'Splits', 'The layer point' and 'Rotate'.
//!
//! The leaf stage is the gadgets' (#118). Here its claims are true values at random points, which
//! both sides bind under a test label, as the trees' messages will. The verifier ends with the
//! pool's values, and each must equal its committed tensors' extensions, computed from the witness.
//!
//! A graph without layers (RFC-0072 'Graphs without layers'), with no checkpoint and no logits,
//! proves with the same relations through a prover of the graph alone, and a committed input
//! changed under the honest values is rejected.
//!
//! The verifier holds the public tensors, which it computes from the public graph inputs alone
//! (RFC-0072 'Public tensors'). It moves each claim's public terms to the claim's value before its
//! split check, evaluates a `ConstAdd`'s shift `ĉ` and reads a relation's public operand itself,
//! and checks a `ConstMul` by a constant per channel as RFC-0010's `Elementwise`. Its evaluation
//! of a public term is the prover's, [`public_value`]; the tests check the shift against an
//! extension of `ψ·c` they enumerate over the padded cube, and that a verifier taking
//! `c̃(r_c)·ψ̃(r)` in its place rejects an honest proof.
//!
//! Stage 1's bit zero-checks (RFC-0072 'The bit zero-checks', #526's second slice) are checked
//! by the verifier's closed forms in [`bits`], which holds their tests. The selection relation,
//! combined selections and `BitFn` (RFC-0072 'The selection relation', 'Combined selections';
//! RFC-0066 '`BitFn`', #526's third slice) are checked by the closed forms in [`selections`],
//! which holds theirs. [`sha256`] proves the SHA-256 circuit end to end through all of them.
//!
//! #526's fourth slice: [`padding`] proves each operation over the padded tables of a lookup
//! non-zero at padding (RFC-0072 'Padded positions'); [`lookups`] proves a graph's lookups
//! through the real leaf stage, checked by a verifier of LogUp and RFC-0045's split that
//! evaluates public lookup inputs itself, then by the stage verifier here on the same transcript
//! ([`verify_from_leaf`]); and [`aes`] proves AES-128-CTR end to end through all of it.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use tessera_codec::{Ascii, DecodeLimits, Secret, SecretBuf, envelope, schema};
use tessera_compiler::schedule::LayerMap;
use tessera_compiler::schedule::public_tensors;
use tessera_compiler::schedule::{
    Claim, Coord, Factor, Layers, NodeRef, Part, PublicSource, Relation, Source, StageSchedule,
};
use tessera_compiler::{
    CompileError, Lowering, PublicModel, derive_graph_schedule, derive_schedule, lower,
};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{
    CommitmentClass, Graph, GraphBuilder, IntRange, IntType, IntegerModel, NodeKind, Op,
    TensorDecl, TensorId, WeightRef, WeightType,
};
use tessera_poly::{eq, eq_table, evaluate_base, less_than};
use tessera_prover::leaf::gadget_nodes;
use tessera_prover::{
    PaddedWitness, PublicTensors, StageError, StageProof, StageProver, StageStateMemory,
    WitnessSource, public_value,
};
use tessera_sumcheck::SumcheckTranscript;
use tessera_tables::{GeneratedTable, RopeTable, Table, TableId, TableKind, TableSpec};
use tessera_witness::{Engine, Witness};

type F = GoldilocksExt2;

mod aes;
mod bits;
mod lookups;
mod padding;
mod selections;
mod sha256;

// ----- Models ---------------------------------------------------------------------------------

/// A conformance model of `spec/vectors/models/` with its first sequence's tokens.
fn conformance(name: &str) -> (IntegerModel, Vec<u32>) {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models");
    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("models.json")).expect("the index"))
            .expect("JSON");
    let entry = index["models"]
        .as_array()
        .expect("models")
        .iter()
        .find(|m| m["name"] == name)
        .expect("the model");
    let bytes = fs::read(dir.join(entry["file"].as_str().expect("a file"))).expect("the file");
    let model = envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
        .expect("an IntegerModel");
    let tokens = entry["sequences"][0]["tokens"]
        .as_array()
        .expect("tokens")
        .iter()
        .map(|t| u32::try_from(t.as_u64().expect("a token")).expect("small"))
        .collect();
    (model, tokens)
}

/// Every conformance model: each rescales every layer's sites alike (RFC-0038), so its layers
/// lower to one template.
const MODELS: [&str; 4] = [
    "tiny-2-servable",
    "tiny-2-minimal",
    "tiny-2-affine",
    "tiny-4-servable",
];

/// A model lowered for its tokens, its schedule, and its witness.
fn setup(name: &str) -> (Lowering, StageSchedule, Witness) {
    let (model, tokens) = conformance(name);
    let n = u32::try_from(tokens.len()).expect("small");
    let lowering = lower(PublicModel::of(&model), n).expect("the model lowers");
    let schedule = derive_schedule(&lowering).expect("a schedule");
    let witness = Engine::new(&lowering)
        .expect("an engine")
        .run(&model, &Secret::new(tokens))
        .expect("a witness");
    (lowering, schedule, witness)
}

// ----- Proofs ---------------------------------------------------------------------------------

/// What the stages send: the leaf stage's claims and splits (the trees' are #118's), then each
/// reduction stage's messages.
#[derive(Clone, Debug)]
struct Proof {
    leaf: Vec<Vec<F>>,
    leaf_splits: Vec<F>,
    stages: Vec<StageProof>,
}

fn transcript() -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: [0x13; 32],
        session_id: [0x37; 16],
    })
}

/// The leaf stage's claims, bound as the trees' messages will bind them.
fn absorb_leaf(transcript: &mut Transcript, leaf: &[Vec<F>]) {
    let flat: Vec<F> = leaf.iter().flatten().copied().collect();
    transcript.absorb_message("test/leaf", &flat);
}

/// Random leaf points, one per site, of its claim's coordinates.
fn leaf_points(schedule: &StageSchedule, seed: u64) -> Vec<Vec<F>> {
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Goldilocks::from_u64_reduced(z ^ (z >> 31))
    };
    schedule.stages[0]
        .claims
        .iter()
        .map(|claim| {
            (0..claim.point.len())
                .map(|_| {
                    let c0 = next();
                    F::new(c0, next())
                })
                .collect()
        })
        .collect()
}

/// Every stage proven, and the pool the prover ends with.
fn prove(
    lowering: &Lowering,
    schedule: &StageSchedule,
    witness: &Witness,
    points: &[Vec<F>],
) -> (Proof, Vec<Option<F>>) {
    let mut t = transcript();
    let mut prover = StageProver::new(lowering, schedule, witness);
    let leaf = prover.leaf(points).expect("the leaf stage's claims");
    absorb_leaf(&mut t, &leaf);
    let leaf_splits = prover.leaf_splits(&mut t).expect("the leaf stage's splits");
    let stages = (1..schedule.stages.len())
        .map(|k| {
            let k = u16::try_from(k).expect("small");
            prover.prove_stage(&mut t, k).expect("a stage")
        })
        .collect();
    let proof = Proof {
        leaf,
        leaf_splits,
        stages,
    };
    (proof, prover.pool().to_vec())
}

/// Every stage of a schedule of `graph` proven by a prover of the graph alone, over `witness`.
fn prove_graph<W: WitnessSource + ?Sized>(
    graph: &Graph,
    tables: &[TableSpec],
    schedule: &StageSchedule,
    witness: &W,
    points: &[Vec<F>],
) -> (Proof, Vec<Option<F>>) {
    let mut t = transcript();
    let mut prover = StageProver::for_graph(graph, tables, schedule, witness);
    let leaf = prover.leaf(points).expect("the leaf stage's claims");
    absorb_leaf(&mut t, &leaf);
    let leaf_splits = prover.leaf_splits(&mut t).expect("the leaf stage's splits");
    let stages = (1..schedule.stages.len())
        .map(|k| {
            let k = u16::try_from(k).expect("small");
            prover.prove_stage(&mut t, k).expect("a stage")
        })
        .collect();
    let proof = Proof {
        leaf,
        leaf_splits,
        stages,
    };
    (proof, prover.pool().to_vec())
}

// ----- A verifier from RFC-0010 and RFC-0037 ----------------------------------------------------

/// `Reject(Sumcheck { stage, round })` (04 §2), the leaf stage's splits at stage 0, round 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Reject {
    stage: u16,
    round: u32,
}

/// How a verifier evaluates a `ConstAdd`'s shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shift {
    /// RFC-0072's `ĉ`, the extension of `ψ·c`: [`public_value`].
    Extension,
    /// `c̃(r_c)·ψ̃(r)`, which agrees with `ĉ` on the cube only: a wrong verifier.
    ChannelsTimesSelector,
    /// None: a wrong verifier that ignores the shift.
    Omitted,
}

/// How a verifier evaluates a selection's kernel (RFC-0072 'The selection relation').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    /// RFC-0072's `K̃(r_a, r'_k) = Σ_t a_t·S̃_t(r_a, r'_k)`, each member's by its closed form.
    Closed,
    /// The same with the last member's rows each moved one row down its axis: a wrong verifier.
    LastMemberMoved,
}

/// What a verifier holds beside the proof: the graph's public tensors, which it computes from
/// public data, and how it evaluates a shift and a selection's kernel.
#[derive(Clone, Copy)]
struct Publics<'a> {
    values: &'a dyn WitnessSource,
    shift: Shift,
    kernel: Kernel,
}

/// A graph with no public tensor but its inputs, which no claim reaches: the Llama templates'.
struct NoPublic;

impl WitnessSource for NoPublic {
    fn values(&self, _: TensorId) -> Option<&[i64]> {
        None
    }
}

const NO_PUBLIC: Publics<'static> = Publics {
    values: &NoPublic,
    shift: Shift::Extension,
    kernel: Kernel::Closed,
};

/// The output tensor -> producer node map of `graph`.
fn producers_of(graph: &Graph) -> Vec<Option<tessera_ir::NodeId>> {
    let mut producers = vec![None; graph.tensors.len()];
    for node in &graph.nodes {
        for &out in &node.outputs {
            producers[out.0 as usize] = Some(node.id);
        }
    }
    producers
}

type PublicCacheKey = (PublicSource, Vec<F>);
type KernelCacheKey = (tessera_ir::NodeId, Vec<F>, Vec<F>);

/// The verifier's state: the graph and its tables, its public data, the challenges, each stage's
/// claims' and parts' values, and the pool.
struct Verifier<'a> {
    graph: &'a Graph,
    tables: &'a [TableSpec],
    schedule: &'a StageSchedule,
    publics: Publics<'a>,
    /// The graph's public tensors (RFC-0072 'Public tensors').
    public: PublicTensors,
    producers: Vec<Option<tessera_ir::NodeId>>,
    /// RFC-0072 'Once per point': public terms and selection kernels evaluated once per point.
    public_cache: std::cell::RefCell<HashMap<PublicCacheKey, F>>,
    kernel_cache: std::cell::RefCell<HashMap<KernelCacheKey, F>>,
    coords: HashMap<Coord, F>,
    claims: Vec<Vec<Vec<F>>>,
    parts: Vec<Vec<Vec<Vec<F>>>>,
    pool: Vec<Option<F>>,
}

/// The pool's values, once every stage's checks pass.
fn verify(
    lowering: &Lowering,
    schedule: &StageSchedule,
    proof: &Proof,
    points: &[Vec<F>],
) -> Result<Vec<F>, Reject> {
    verify_graph(&lowering.graph, &lowering.tables, schedule, proof, points)
}

/// The pool's values of a schedule of `graph`, whose tables are `tables`, once every stage's
/// checks pass.
fn verify_graph(
    graph: &Graph,
    tables: &[TableSpec],
    schedule: &StageSchedule,
    proof: &Proof,
    points: &[Vec<F>],
) -> Result<Vec<F>, Reject> {
    verify_public(graph, tables, schedule, proof, points, NO_PUBLIC)
}

/// The pool's values of a schedule of `graph` whose verifier holds `publics`, once every stage's
/// checks pass.
fn verify_public(
    graph: &Graph,
    tables: &[TableSpec],
    schedule: &StageSchedule,
    proof: &Proof,
    points: &[Vec<F>],
    publics: Publics<'_>,
) -> Result<Vec<F>, Reject> {
    let mut v = Verifier {
        graph,
        tables,
        schedule,
        publics,
        public: PublicTensors::of(graph),
        producers: producers_of(graph),
        public_cache: std::cell::RefCell::new(HashMap::new()),
        kernel_cache: std::cell::RefCell::new(HashMap::new()),
        coords: HashMap::new(),
        claims: Vec::new(),
        parts: Vec::new(),
        pool: vec![None; schedule.pool.len()],
    };
    let mut t = transcript();
    for (s, point) in (0u32..).zip(points) {
        for (index, &x) in (0u32..).zip(point) {
            v.coords.insert(Coord::Leaf { site: s, index }, x);
        }
    }
    let reject = |stage, round| Reject { stage, round };
    let leaf = &schedule.stages[0];
    let shape_ok = proof.leaf.len() == leaf.claims.len()
        && proof
            .leaf
            .iter()
            .zip(&leaf.claims)
            .all(|(values, claim)| values.len() == v.layers_of(claim).len());
    if !shape_ok {
        return Err(reject(0, 0));
    }
    v.claims.push(proof.leaf.clone());
    absorb_leaf(&mut t, &proof.leaf);
    v.splits(0, &proof.leaf_splits, &mut t)?;
    // With no reduction stage, no stage 1 stacks the leaf stage's pool claims at `ρ_1`.
    if schedule.stages.len() == 1 {
        v.fill_pool(0);
    }
    if proof.stages.len() + 1 != schedule.stages.len() {
        return Err(reject(u16::try_from(proof.stages.len()).unwrap_or(0), 0));
    }
    for (i, stage) in proof.stages.iter().enumerate() {
        let k = u16::try_from(i + 1).expect("small");
        if stage.stage != k {
            return Err(reject(k, 0));
        }
        v.stage(k, stage, &mut t)?;
    }
    v.pool
        .iter()
        .map(|value| value.ok_or(reject(0, 0)))
        .collect()
}

/// What a verifier holds when it has checked a real leaf stage itself (RFC-0045, RFC-0008): the
/// leaf point `P`, each leaf claim's values, which it derived from the split's values or from
/// public data, and the graph atoms' split values, absorbed with the lookups' under
/// `stage/0/split`.
struct LeafOutcome {
    point: Vec<F>,
    claims: Vec<Vec<F>>,
    atoms: Vec<F>,
}

/// The pool's values and the verifier's coordinates of a schedule of `graph` whose verifier holds
/// `publics`, once every stage's checks pass after the leaf stage it checked itself on `t`
/// ([`LeafOutcome`]): each leaf claim at its site's prefix of `P`, its atoms' splits checked
/// (their message already absorbed), then each reduction stage on the same transcript.
fn verify_from_leaf<T: SumcheckTranscript<F> + ?Sized>(
    graph: &Graph,
    tables: &[TableSpec],
    schedule: &StageSchedule,
    publics: Publics<'_>,
    leaf: LeafOutcome,
    stages: &[StageProof],
    t: &mut T,
) -> Result<(Vec<F>, HashMap<Coord, F>), Reject> {
    let mut v = Verifier {
        graph,
        tables,
        schedule,
        publics,
        public: PublicTensors::of(graph),
        producers: producers_of(graph),
        public_cache: std::cell::RefCell::new(HashMap::new()),
        kernel_cache: std::cell::RefCell::new(HashMap::new()),
        coords: HashMap::new(),
        claims: Vec::new(),
        parts: Vec::new(),
        pool: vec![None; schedule.pool.len()],
    };
    let reject = |stage, round| Reject { stage, round };
    let claims = &schedule.stages[0].claims;
    for ((s, claim), values) in (0u32..).zip(claims).zip(&leaf.claims) {
        for (index, c) in (0u32..).zip(&claim.point) {
            let &x = leaf.point.get(index as usize).ok_or(reject(0, 0))?;
            assert_eq!(*c, Coord::Leaf { site: s, index }, "a leaf claim's point");
            v.coords.insert(Coord::Leaf { site: s, index }, x);
        }
        if values.len() != v.layers_of(claim).len() {
            return Err(reject(0, 0));
        }
    }
    if leaf.claims.len() != claims.len() {
        return Err(reject(0, 0));
    }
    v.claims.push(leaf.claims);
    v.check_splits(0, &leaf.atoms)?;
    if schedule.stages.len() == 1 {
        v.fill_pool(0);
    }
    if stages.len() + 1 != schedule.stages.len() {
        return Err(reject(u16::try_from(stages.len()).unwrap_or(0), 0));
    }
    for (i, stage) in stages.iter().enumerate() {
        let k = u16::try_from(i + 1).expect("small");
        if stage.stage != k {
            return Err(reject(k, 0));
        }
        v.stage(k, stage, t)?;
    }
    let pool = v
        .pool
        .iter()
        .map(|value| value.ok_or(reject(0, 0)))
        .collect::<Result<_, _>>()?;
    Ok((pool, v.coords))
}

impl Verifier<'_> {
    fn coord(&self, c: Coord) -> F {
        if c == Coord::Half {
            return F::from(Goldilocks::from_u64_reduced(2))
                .inverse()
                .expect("2 ≠ 0");
        }
        *self
            .coords
            .get(&c)
            .expect("a coordinate drawn before it is read")
    }

    fn point(&self, coords: &[Coord]) -> Vec<F> {
        coords.iter().map(|&c| self.coord(c)).collect()
    }

    fn layers_of(&self, claim: &Claim) -> Vec<Option<u32>> {
        match claim.layers {
            Layers::Each => (0..self.schedule.layers.count).map(Some).collect(),
            _ => vec![None],
        }
    }

    /// The constant of a `ConstMul` by one constant, in `layer` at a template position.
    fn constant(&self, node: NodeRef, layer: u32) -> F {
        let id = match node {
            NodeRef::Node(id) => id,
            NodeRef::Position(position) => self.schedule.layers.node(position, layer),
        };
        let Op::ConstMul { c } = self.graph.nodes[id.0 as usize].op else {
            panic!("a ConstMul");
        };
        F::from(Goldilocks::from_i64(self.graph.constants[c.0 as usize][0]))
    }

    /// A coefficient in `layer`.
    fn coefficient(&self, products: &[Vec<Factor>], layer: u32) -> F {
        products
            .iter()
            .map(|product| {
                product.iter().fold(F::ONE, |value, factor| {
                    value
                        * match factor {
                            Factor::Minus => -F::ONE,
                            Factor::Pow2(j) => F::from(Goldilocks::from_u64_reduced(1 << j)),
                            Factor::Constant(node) => self.constant(*node, layer),
                            Factor::Lt { coords, len } => {
                                less_than(&self.point(coords), u64::from(*len)).expect("LT")
                            }
                        }
                })
            })
            .fold(F::ZERO, |s, x| s + x)
    }

    /// `Σ_ℓ eq(ρ_1, ℓ)·v_ℓ` over the real layers (RFC-0037 'The layer point').
    fn stack(&self, values: &[F]) -> F {
        let rho: Vec<F> = (0..self.schedule.layers.vars)
            .map(|c| self.coord(Coord::Layer(c)))
            .collect();
        (0u64..)
            .zip(values)
            .map(|(l, &v)| eq_at(&rho, l) * v)
            .fold(F::ZERO, |s, x| s + x)
    }

    /// The value at `point` of a public term's source, which the verifier computes from public
    /// data (RFC-0072 'Public tensors'), a shift by the verifier's rule.
    fn public_term(&self, source: PublicSource, point: &[F]) -> F {
        match (source, self.publics.shift) {
            (PublicSource::Shift(_), Shift::Omitted) => F::ZERO,
            (PublicSource::Shift(node), Shift::ChannelsTimesSelector) => {
                channels_times_selector(self.graph, node, point)
            }
            _ => {
                let key = (source, point.to_vec());
                if let Some(&cached) = self.public_cache.borrow().get(&key) {
                    return cached;
                }
                let val =
                    public_value(self.graph, &self.public, source, self.publics.values, point)
                        .expect("the verifier's public values");
                self.public_cache.borrow_mut().insert(key, val);
                val
            }
        }
    }

    /// The sum of a claim's public terms, which move to its value (RFC-0037 'Splits').
    fn public_sum(&self, claim: &Claim) -> F {
        claim.public.iter().fold(F::ZERO, |sum, term| {
            let value = self.public_term(term.source, &self.point(&term.point));
            sum + self.coefficient(&term.coefficient, 0) * value
        })
    }

    /// The value of input `i` of `node` at `point` in a relation's final check: a public input's
    /// the verifier computes (RFC-0072 'Public tensors' 'As an operand'), any other's the next of
    /// the instance's evaluations `e`.
    fn read(&self, node: &tessera_ir::Node, i: usize, point: &[F], e: &[F], next: &mut usize) -> F {
        let tensor = node.inputs[i];
        if self.public.contains(tensor) {
            return self.public_term(PublicSource::Tensor(tensor), point);
        }
        *next += 1;
        e[*next - 1]
    }

    /// The splits of stage `index` (RFC-0037 'Splits'), checked, then absorbed under
    /// `stage/{index}/split`.
    fn splits<T: SumcheckTranscript<F> + ?Sized>(
        &mut self,
        index: usize,
        values: &[F],
        t: &mut T,
    ) -> Result<(), Reject> {
        self.check_splits(index, values)?;
        t.absorb_message(&format!("stage/{index}/split"), values);
        Ok(())
    }

    /// The checks of stage `index`'s splits (RFC-0037 'Splits'): each claim's public terms move to
    /// its value; then each split claim's parts' values must add up to the rest, layer by layer
    /// for a leaf claim in the layers, and a claim with no part must have no rest.
    fn check_splits(&mut self, index: usize, values: &[F]) -> Result<(), Reject> {
        let stage = &self.schedule.stages[index];
        let fail = Reject {
            stage: u16::try_from(index).expect("small"),
            round: stage.rounds,
        };
        let mut next = 0;
        let mut parts = Vec::with_capacity(stage.claims.len());
        for (c, claim) in stage.claims.iter().enumerate() {
            let layers = self.layers_of(claim);
            let mut per_part = vec![Vec::new(); claim.parts.len()];
            let public = self.public_sum(claim);
            for (l, layer) in layers.iter().enumerate() {
                let value = self.claims[index][c][l] - public;
                if claim.parts.is_empty() {
                    if value != F::ZERO {
                        return Err(fail);
                    }
                    continue;
                }
                if !claim.split {
                    per_part[0].push(value);
                    continue;
                }
                let mut sum = F::ZERO;
                for (p, part) in claim.parts.iter().enumerate() {
                    let sent = *values.get(next).ok_or(fail)?;
                    next += 1;
                    sum += match part {
                        Part::Instance { atom, .. } => {
                            let atom = &claim.atoms[*atom];
                            self.coefficient(&atom.coefficient, layer.unwrap_or(0)) * sent
                        }
                        Part::Pool { .. } => sent,
                    };
                    per_part[p].push(sent);
                }
                if sum != value {
                    return Err(fail);
                }
            }
            parts.push(per_part);
        }
        if next != values.len() {
            return Err(fail);
        }
        self.parts.push(parts);
        // The leaf stage's committed parts are stacked at ρ_1, which stage 1 draws.
        if index > 0 {
            self.fill_pool(index);
        }
        Ok(())
    }

    /// The pool's values of stage `index`'s committed parts, a leaf claim's stacked at `ρ_1`.
    fn fill_pool(&mut self, index: usize) {
        let stage = &self.schedule.stages[index];
        for (claim, per_part) in stage.claims.iter().zip(&self.parts[index]) {
            for (part, layer_values) in claim.parts.iter().zip(per_part) {
                if let Part::Pool { claim: pool, .. } = part {
                    self.pool[*pool] = Some(if claim.layers == Layers::Each {
                        self.stack(layer_values)
                    } else {
                        layer_values[0]
                    });
                }
            }
        }
    }

    /// Stage `k`: its points, its batched sumcheck checked round by round and at the end, its
    /// claims, then its splits.
    fn stage<T: SumcheckTranscript<F> + ?Sized>(
        &mut self,
        k: u16,
        proof: &StageProof,
        t: &mut T,
    ) -> Result<(), Reject> {
        let stage = &self.schedule.stages[usize::from(k)];
        if k == 1 {
            for c in 0..self.schedule.layers.vars {
                let rho = t.challenge(&format!("stage/1/layer/{c}"));
                self.coords.insert(Coord::Layer(c), rho);
            }
            for c in 0..stage.zero_vars {
                let z = t.challenge(&format!("stage/1/zero/{c}"));
                self.coords.insert(Coord::Zero(c), z);
            }
            self.fill_pool(0);
        }
        // The instances' claims.
        let claims: Vec<F> = stage
            .instances
            .iter()
            .map(|instance| match instance.source {
                Source::Part { claim, part } => {
                    let values = &self.parts[usize::from(k) - 1][claim][part];
                    let leaf = &self.schedule.stages[usize::from(k) - 1].claims[claim];
                    if leaf.layers == Layers::Each {
                        self.stack(values)
                    } else {
                        values[0]
                    }
                }
                Source::ZeroCheck => F::ZERO,
                Source::Native(_) => panic!("this oracle covers graph-only schedules"),
            })
            .collect();
        let degree = stage
            .instances
            .iter()
            .map(|i| self.degree(i))
            .max()
            .unwrap_or(1);
        let n_max = stage.rounds;
        let gamma = t.challenge(&format!("stage/{k}/gamma"));
        let mut claim = F::ZERO;
        let mut power = F::ONE;
        for (instance, &c) in stage.instances.iter().zip(&claims) {
            let scale = F::from(Goldilocks::from_u64_reduced(1 << (n_max - instance.vars)));
            claim += power * scale * c;
            power *= gamma;
        }
        let rounds = &proof.sumcheck.rounds;
        for j in 0..n_max {
            let message = rounds
                .get(j as usize)
                .ok_or(Reject { stage: k, round: j })?;
            if message.len() != degree {
                return Err(Reject { stage: k, round: j });
            }
            t.absorb_message(&format!("stage/{k}/round/{j}"), message);
            let r = t.challenge(&format!("stage/{k}/r/{j}"));
            self.coords.insert(Coord::Round { stage: k, round: j }, r);
            claim = round_value(claim, message, r);
        }
        let end = Reject {
            stage: k,
            round: n_max,
        };
        if rounds.len() != n_max as usize {
            return Err(end);
        }
        let evals = &proof.sumcheck.final_evals;
        let expected: usize = stage.instances.iter().map(|i| i.claims.len()).sum();
        if evals.len() != expected {
            return Err(end);
        }
        t.absorb_message(&format!("stage/{k}/evals"), evals);
        let mut at = 0;
        let mut batched = F::ZERO;
        let mut power = F::ONE;
        let mut values = vec![Vec::new(); stage.claims.len()];
        for instance in &stage.instances {
            let e = &evals[at..at + instance.claims.len()];
            at += instance.claims.len();
            batched += power * self.relation(k, instance, e);
            power *= gamma;
            for (c, &value) in instance.claims.clone().zip(e) {
                values[c] = vec![value];
            }
        }
        if batched != claim {
            return Err(end);
        }
        self.claims.push(values);
        self.splits(usize::from(k), &proof.splits, t)
    }

    /// An instance's degree (RFC-0010 'Relations', RFC-0037 'Rotate'): its factors, `eq` among
    /// them.
    fn degree(&self, instance: &tessera_compiler::schedule::Instance) -> usize {
        let stacked = matches!(instance.layers, Layers::Eq(_)) && self.schedule.layers.vars > 0;
        let node = self.node(instance.node);
        match instance.relation {
            Relation::Native { .. } => panic!("this oracle covers graph-only schedules"),
            Relation::MatMulWeight => 2 + usize::from(stacked),
            Relation::MatMulAct { .. } => {
                let heads = self.graph.tensors[node.inputs[0].0 as usize].shape[0];
                2 + usize::from(stacked || heads > 1)
            }
            Relation::Mul(_)
            | Relation::SumSquares
            | Relation::ScaleChannels
            | Relation::Booleanity
            | Relation::OneHotVocabulary => 3,
            Relation::OneHotCount | Relation::OneHotIndex => 2,
            Relation::Rotate => 2,
            // The kernel times `x` (RFC-0072 'The selection relation').
            Relation::Selection => 2,
            // `eq` times RFC-0066's polynomial, of degree 2 or 3.
            Relation::BitFn(f) => match f {
                tessera_ir::BoolFn::Xor3 | tessera_ir::BoolFn::Maj => 4,
                tessera_ir::BoolFn::Xor | tessera_ir::BoolFn::And | tessera_ir::BoolFn::Ch => 3,
            },
            // RFC-0072's table: the three checks of degree 3, recomposition's of 2.
            Relation::InputPadding | Relation::BitPadding | Relation::AssertEqual => 3,
            Relation::Recomposition => 2,
            Relation::Equality => match node.op {
                Op::Mul { .. } => 3,
                // `c̃(x_c)·X̃(x)`, the constant per channel a public factor.
                Op::ConstMul { c } if self.graph.constants[c.0 as usize].len() > 1 => 3,
                Op::ConstMul { .. } => match instance.node {
                    NodeRef::Position(p) if !self.schedule.layers.uniform(p) => 3,
                    _ => 2,
                },
                _ => 2,
            },
        }
    }

    fn node(&self, node: NodeRef) -> &tessera_ir::Node {
        let id = match node {
            NodeRef::Node(id) => id,
            NodeRef::Position(p) => self.schedule.layers.node(p, 0),
        };
        &self.graph.nodes[id.0 as usize]
    }

    /// An instance of stage `k`'s relation at its point from its evaluations, public factors
    /// included.
    fn relation(&self, k: u16, instance: &tessera_compiler::schedule::Instance, e: &[F]) -> F {
        let mut target = self.point(&instance.point);
        let stacked = matches!(instance.layers, Layers::Eq(_));
        let lambda = if stacked {
            self.schedule.layers.vars as usize
        } else {
            0
        };
        if let Layers::Eq(rho) = &instance.layers {
            target.extend(self.point(rho));
        }
        let r = self.point(&instance.challenges);
        let node = self.node(instance.node);
        let shape = |i: usize| &self.graph.tensors[node.inputs[i].0 as usize].shape;
        match instance.relation {
            Relation::Native { .. } => panic!("this oracle covers graph-only schedules"),
            Relation::MatMulWeight => {
                let layers =
                    eq(&target[target.len() - lambda..], &r[r.len() - lambda..]).expect("eq");
                layers * e[0] * e[1]
            }
            Relation::MatMulAct { .. } => {
                let out = &self.graph.tensors[node.outputs[0].0 as usize].shape;
                let (c, a) = (axis_vars(out[2]) as usize, axis_vars(out[1]) as usize);
                let inner = axis_vars(shape(0)[2]) as usize;
                eq(&target[c + a..], &r[inner..]).expect("eq") * e[0] * e[1]
            }
            // A public operand the verifier reads itself, a broadcast one at its own coordinates.
            Relation::Mul(broadcast) => {
                let b = broadcast_coords(&r, shape(0), shape(1), broadcast);
                let mut next = 0;
                let a = self.read(node, 0, &r, e, &mut next);
                let b = self.read(node, 1, &b, e, &mut next);
                eq(&target, &r).expect("eq") * a * b
            }
            Relation::SumSquares => {
                let last = axis_vars(*shape(0).last().expect("an axis")) as usize;
                eq(&target, &r[last..]).expect("eq") * e[0] * e[0]
            }
            Relation::Rotate => self.kernel(node, &target, &r, stacked) * e[0],
            // RFC-0010's `Elementwise` with the constant per channel a public operand, 0 at padded
            // channels (RFC-0072 'ConstMul by a constant per channel').
            Relation::ScaleChannels => {
                eq(&target, &r).expect("eq") * self.channels(node, &r) * e[0]
            }
            Relation::Booleanity => eq(&target, &r).expect("eq") * e[0] * (F::ONE - e[0]),
            Relation::OneHotVocabulary | Relation::OneHotCount | Relation::OneHotIndex => {
                let &[positions, vocab] = self.graph.tensors[node.outputs[0].0 as usize]
                    .shape
                    .as_slice()
                else {
                    panic!("OneHot axes")
                };
                let vbits = axis_vars(vocab) as usize;
                let rv = &r[..vbits];
                let rows = &r[vbits..];
                let scale = F::from(Goldilocks::from_u64_reduced(
                    vocab.next_power_of_two() as u64
                ))
                .inverse()
                .unwrap();
                match instance.relation {
                    Relation::Native { .. } => panic!("this oracle covers graph-only schedules"),
                    Relation::OneHotVocabulary => {
                        eq(&target, &r).unwrap()
                            * (F::ONE - less_than(rv, vocab as u64).unwrap())
                            * e[0]
                    }
                    Relation::OneHotCount => {
                        eq(&target[vbits..], rows).unwrap()
                            * (e[0] - less_than(rows, positions as u64).unwrap() * scale)
                    }
                    Relation::OneHotIndex => {
                        let index = rv.iter().enumerate().fold(F::ZERO, |sum, (i, &x)| {
                            sum + F::from(Goldilocks::from_u64_reduced(1 << i)) * x
                        });
                        eq(&target[vbits..], rows).unwrap() * (index * e[0] - e[1] * scale)
                    }
                    _ => unreachable!(),
                }
            }
            Relation::Equality => {
                // `f`'s inputs that are not public are the evaluations after `Y`'s.
                let mut next = 1;
                let f = match node.op {
                    Op::ConstMul { c } if self.graph.constants[c.0 as usize].len() > 1 => {
                        self.channels(node, &r) * self.read(node, 0, &r, e, &mut next)
                    }
                    // `f = x̃ + ĉ` (RFC-0072 'A committed output').
                    Op::ConstAdd { .. } => {
                        self.read(node, 0, &r, e, &mut next)
                            + self.public_term(PublicSource::Shift(node.id), &r)
                    }
                    Op::ConstMul { .. } => {
                        // The constant of the instance point's layer: a public factor.
                        let c = match instance.node {
                            NodeRef::Position(p) if stacked => {
                                let rho = &r[r.len() - lambda..];
                                (0..self.schedule.layers.count)
                                    .map(|l| {
                                        eq_at(rho, u64::from(l))
                                            * self.constant(NodeRef::Position(p), l)
                                    })
                                    .fold(F::ZERO, |s, x| s + x)
                            }
                            other => self.constant(other, 0),
                        };
                        c * self.read(node, 0, &r, e, &mut next)
                    }
                    Op::Add | Op::Sub => {
                        let a = self.read(node, 0, &r, e, &mut next);
                        let b = self.read(node, 1, &r, e, &mut next);
                        if node.op == Op::Add { a + b } else { a - b }
                    }
                    Op::Mul { broadcast } => {
                        let b = broadcast_coords(&r, shape(0), shape(1), broadcast);
                        let a = self.read(node, 0, &r, e, &mut next);
                        a * self.read(node, 1, &b, e, &mut next)
                    }
                    _ => panic!("an equality zero-check's operation"),
                };
                eq(&target, &r).expect("eq") * (e[0] - f)
            }
            Relation::Selection => self.selection_relation(k, instance, node, &target, &r, e),
            Relation::BitFn(f) => {
                let mut next = 0;
                let inputs: Vec<F> = (0..node.inputs.len())
                    .map(|i| self.read(node, i, &r, e, &mut next))
                    .collect();
                eq(&target, &r).expect("eq") * selections::bit_function(f, &inputs)
            }
            Relation::InputPadding
            | Relation::BitPadding
            | Relation::Recomposition
            | Relation::AssertEqual => self.bit_relation(instance, node, &target, &r, e),
        }
    }

    /// `c̃(r_c)` of a `ConstMul` by a constant per channel at the channel coordinates of `r`, the
    /// lowest, the constant 0 at padded channels.
    fn channels(&self, node: &tessera_ir::Node, r: &[F]) -> F {
        let Op::ConstMul { c } = node.op else {
            panic!("a ConstMul");
        };
        let out = &self.graph.tensors[node.outputs[0].0 as usize].shape;
        let low = axis_vars(*out.last().expect("a channel axis")) as usize;
        constant_extension(&self.graph.constants[c.0 as usize], &r[..low])
    }

    /// RoPE's kernel at the instance's point `r'` for the claim at `r` (RFC-0037 'Rotate'):
    /// `eq(r_b, r'_b)·⟨C, w⟩ + (r_b − r'_b)·⟨S, w⟩`, `w = eq(r_{−b}, ·)·eq(r'_{−b}, ·)`, the
    /// constants the row's position's and the frequency's, every head's and layer's, 0 on a
    /// padded row.
    fn kernel(&self, node: &tessera_ir::Node, r: &[F], r2: &[F], stacked: bool) -> F {
        let Op::Rotate { rope } = node.op else {
            panic!("a Rotate");
        };
        let spec = self
            .tables
            .iter()
            .find(|s| s.id == rope)
            .expect("the table");
        let table = RopeTable::from_spec(spec).expect("RoPE's constants");
        let x = &self.graph.tensors[node.inputs[0].0 as usize].shape;
        let (rows, heads, dim) = (x[0], x[1], x[2]);
        let bit = axis_vars(dim) as usize - 1;
        let (hv, nv) = (axis_vars(heads) as usize, axis_vars(rows) as usize);
        let lv = if stacked {
            self.schedule.layers.vars as usize
        } else {
            0
        };
        let without = |p: &[F]| -> Vec<F> {
            p.iter()
                .enumerate()
                .filter(|&(k, _)| k != bit)
                .map(|(_, &v)| v)
                .collect()
        };
        let (w1, w2) = (
            eq_table(&without(r)).expect("eq"),
            eq_table(&without(r2)).expect("eq"),
        );
        let (mut cos, mut sin) = (F::ZERO, F::ZERO);
        for k in 0..1usize << (bit + hv + nv + lv) {
            let row = (k >> (bit + hv)) & ((1 << nv) - 1);
            if row >= rows as usize {
                continue;
            }
            let frequency = u32::try_from(k & ((1 << bit) - 1)).expect("small");
            let position = u32::try_from(row).expect("small");
            let (c, s) = table.constants().at(position, frequency).expect("a pair");
            let w = w1[k] * w2[k];
            cos += w * F::from(Goldilocks::from_i64(c));
            sin += w * F::from(Goldilocks::from_i64(s));
        }
        eq(&[r[bit]], &[r2[bit]]).expect("eq") * cos + (r[bit] - r2[bit]) * sin
    }
}

/// `p(r)` from a round message `p(0), p(2), …, p(d)` and the claim `p(0) + p(1)`, by Lagrange's
/// formula over the nodes `0, …, d` (RFC-0010 'Sumcheck messages').
fn round_value(claim: F, message: &[F], r: F) -> F {
    let mut values = vec![message[0], claim - message[0]];
    values.extend_from_slice(&message[1..]);
    let d = values.len() - 1;
    let node = |k: usize| F::from(Goldilocks::from_u64_reduced(k as u64));
    (0..=d)
        .map(|k| {
            let (num, den) = (0..=d)
                .filter(|&m| m != k)
                .fold((F::ONE, F::ONE), |(n, d), m| {
                    (n * (r - node(m)), d * (node(k) - node(m)))
                });
            values[k] * num * den.inverse().expect("distinct nodes")
        })
        .fold(F::ZERO, |s, x| s + x)
}

/// The coordinates of `Mul`'s second operand `b` within the point `r` over the first's shape `a`:
/// its highest for `Trailing`, its lowest for `Leading` (RFC-0010 'Relations' 'Broadcast').
fn broadcast_coords(r: &[F], a: &[u32], b: &[u32], broadcast: tessera_ir::Broadcast) -> Vec<F> {
    let vars = |shape: &[u32]| shape.iter().map(|&d| axis_vars(d) as usize).sum::<usize>();
    match broadcast {
        tessera_ir::Broadcast::Trailing => r[vars(a) - vars(b)..].to_vec(),
        tessera_ir::Broadcast::Leading => r[..vars(b)].to_vec(),
    }
}

/// `c̃(r_c)` of a constant per channel at `r_c`, zero at the padded channels.
fn constant_extension(constant: &[i64], r_c: &[F]) -> F {
    let mut table = vec![Goldilocks::ZERO; 1 << r_c.len()];
    for (slot, &v) in table.iter_mut().zip(constant) {
        *slot = Goldilocks::from_i64(v);
    }
    evaluate_base::<F>(&table, r_c).expect("a channel point")
}

/// `c̃(r_c)·ψ̃(r)` of the `ConstAdd` `node`, `ψ̃(r) = Π_a LT(r_a, n_a)` over every axis, the last
/// among them: what a wrong verifier takes for the shift `ĉ`. For a constant of one value it is
/// `c·ψ̃(r) = ĉ(r)`.
fn channels_times_selector(graph: &Graph, node: tessera_ir::NodeId, r: &[F]) -> F {
    let node = &graph.nodes[node.0 as usize];
    let Op::ConstAdd { c } = node.op else {
        panic!("a ConstAdd");
    };
    let constant = &graph.constants[c.0 as usize];
    let shape = &graph.tensors[node.outputs[0].0 as usize].shape;
    let mut selector = F::ONE;
    let mut at = 0;
    for &len in shape.iter().rev() {
        let vars = axis_vars(len) as usize;
        selector *= less_than(&r[at..at + vars], u64::from(len)).expect("LT");
        at += vars;
    }
    let low = axis_vars(*shape.last().expect("an axis")) as usize;
    let c = if constant.len() == 1 {
        F::from(Goldilocks::from_i64(constant[0]))
    } else {
        constant_extension(constant, &r[..low])
    };
    c * selector
}

fn eq_at(rho: &[F], layer: u64) -> F {
    rho.iter().enumerate().fold(F::ONE, |p, (j, &x)| {
        p * if (layer >> j) & 1 == 1 { x } else { F::ONE - x }
    })
}

fn axis_vars(len: u32) -> u32 {
    len.next_power_of_two().trailing_zeros()
}

// ----- The pool against the witness -----------------------------------------------------------

/// `tensor` padded per dimension with zeros.
fn padded(witness: &Witness, graph: &Graph, tensor: TensorId) -> Vec<Goldilocks> {
    let shape = &graph.tensors[tensor.0 as usize].shape;
    let vars: Vec<u32> = shape.iter().map(|&d| axis_vars(d)).collect();
    let mut table = vec![Goldilocks::ZERO; 1 << vars.iter().sum::<u32>()];
    for (flat, &v) in witness.tensor(tensor).expect("a tensor").iter().enumerate() {
        let (mut rest, mut index, mut shift) = (flat, 0usize, 0);
        for (axis, &len) in shape.iter().enumerate().rev() {
            index |= (rest % len as usize) << shift;
            rest /= len as usize;
            shift += vars[axis];
        }
        table[index] = Goldilocks::from_i64(v);
    }
    table
}

/// Each pool claim's value from the witness: its atoms' extensions at its point.
fn pool_from_witness(
    verifier_coords: &HashMap<Coord, F>,
    lowering: &Lowering,
    schedule: &StageSchedule,
    witness: &Witness,
) -> Vec<F> {
    pool_of_graph(verifier_coords, &lowering.graph, schedule, witness)
}

/// Each pool claim's value of a schedule of `graph` from the witness.
fn pool_of_graph(
    verifier_coords: &HashMap<Coord, F>,
    graph: &Graph,
    schedule: &StageSchedule,
    witness: &Witness,
) -> Vec<F> {
    pool_of_tables(verifier_coords, graph, schedule, &|tensor| {
        padded(witness, graph, tensor)
    })
}

/// Each pool claim's value of a schedule of `graph` from the committed tables `table` gives,
/// each tensor's over its padded shape: what the openings check.
fn pool_of_tables(
    verifier_coords: &HashMap<Coord, F>,
    graph: &Graph,
    schedule: &StageSchedule,
    table: &dyn Fn(TensorId) -> Vec<Goldilocks>,
) -> Vec<F> {
    let v = Verifier {
        graph,
        tables: &[],
        schedule,
        publics: NO_PUBLIC,
        public: PublicTensors::of(graph),
        producers: producers_of(graph),
        public_cache: std::cell::RefCell::new(HashMap::new()),
        kernel_cache: std::cell::RefCell::new(HashMap::new()),
        coords: verifier_coords.clone(),
        claims: Vec::new(),
        parts: Vec::new(),
        pool: Vec::new(),
    };
    schedule
        .pool
        .iter()
        .map(|claim| {
            let mut value = F::ZERO;
            for atom in &claim.atoms {
                let weights: Vec<(u32, F)> = match &atom.layers {
                    Layers::None => vec![(0, F::ONE)],
                    Layers::One(l) => vec![(*l, F::ONE)],
                    Layers::Eq(rho) => {
                        let rho = v.point(rho);
                        (0..schedule.layers.count)
                            .map(|l| (l, eq_at(&rho, u64::from(l))))
                            .collect()
                    }
                    Layers::Each => panic!("the pool's atoms are stacked"),
                };
                for (l, w) in weights {
                    let tensor = schedule
                        .layers
                        .tensor(graph, atom.kind, l)
                        .expect("a tensor");
                    let table = table(tensor);
                    let x = evaluate_base::<F>(&table, &v.point(&atom.point)).expect("a point");
                    value += w * v.coefficient(&atom.coefficient, l) * x;
                }
            }
            value
        })
        .collect()
}

/// The verifier's coordinates after a run, for the pool's oracle.
fn coords_after(schedule: &StageSchedule, proof: &Proof, points: &[Vec<F>]) -> HashMap<Coord, F> {
    let mut t = transcript();
    let mut coords = HashMap::new();
    for (s, point) in (0u32..).zip(points) {
        for (index, &x) in (0u32..).zip(point) {
            coords.insert(Coord::Leaf { site: s, index }, x);
        }
    }
    absorb_leaf(&mut t, &proof.leaf);
    t.absorb_message("stage/0/split", &proof.leaf_splits);
    for stage in &proof.stages {
        let k = stage.stage;
        let spec = &schedule.stages[usize::from(k)];
        if k == 1 {
            for c in 0..schedule.layers.vars {
                coords.insert(Coord::Layer(c), t.challenge(&format!("stage/1/layer/{c}")));
            }
            for c in 0..spec.zero_vars {
                coords.insert(Coord::Zero(c), t.challenge(&format!("stage/1/zero/{c}")));
            }
        }
        let _gamma: F = t.challenge(&format!("stage/{k}/gamma"));
        for (j, message) in (0u32..).zip(&stage.sumcheck.rounds) {
            t.absorb_message(&format!("stage/{k}/round/{j}"), message);
            coords.insert(
                Coord::Round { stage: k, round: j },
                t.challenge(&format!("stage/{k}/r/{j}")),
            );
        }
        t.absorb_message(&format!("stage/{k}/evals"), &stage.sumcheck.final_evals);
        t.absorb_message(&format!("stage/{k}/split"), &stage.splits);
    }
    coords
}

// ----- Tests ----------------------------------------------------------------------------------

/// Every stage of every conformance model verifies, and the pool the verifier ends with is the
/// prover's and the committed tensors' extensions (RFC-0037 'Testing Strategy').
#[test]
fn every_stage_of_the_conformance_models_verifies() {
    for name in MODELS {
        let (lowering, schedule, witness) = setup(name);
        let points = leaf_points(&schedule, 0x0113_57A6);
        let (proof, prover_pool) = prove(&lowering, &schedule, &witness, &points);
        let pool = verify(&lowering, &schedule, &proof, &points)
            .unwrap_or_else(|reject| panic!("{name}: rejected at {reject:?}"));
        let prover_pool: Vec<F> = prover_pool
            .into_iter()
            .map(|v| v.expect("a value"))
            .collect();
        assert_eq!(pool, prover_pool, "{name}: the prover's pool");
        let coords = coords_after(&schedule, &proof, &points);
        let oracle = pool_from_witness(&coords, &lowering, &schedule, &witness);
        assert_eq!(
            pool, oracle,
            "{name}: the pool is the committed tensors' extensions"
        );
        assert_eq!(proof.stages.len() + 1, schedule.stage_count(), "{name}");
    }
}

/// A split that does not add up is rejected at its stage, `Reject(Sumcheck { stage: 0, round: 0
/// })` for the leaf stage (RFC-0037 'Splits').
#[test]
fn a_split_that_does_not_add_up_is_rejected_at_its_stage() {
    // Its attention residuals have one class each: their keys `acc + 2^r·res` split.
    let (lowering, schedule, witness) = setup("tiny-4-servable");
    let points = leaf_points(&schedule, 0x5A17);
    let (proof, _) = prove(&lowering, &schedule, &witness, &points);
    assert!(!proof.leaf_splits.is_empty(), "the residual keys split");
    for i in 0..proof.leaf_splits.len() {
        let mut bad = proof.clone();
        bad.leaf_splits[i] += F::ONE;
        assert_eq!(
            verify(&lowering, &schedule, &bad, &points),
            Err(Reject { stage: 0, round: 0 }),
            "split value {i}"
        );
    }
    // One value too many or too few.
    let mut longer = proof.clone();
    longer.leaf_splits.push(F::ONE);
    assert_eq!(
        verify(&lowering, &schedule, &longer, &points),
        Err(Reject { stage: 0, round: 0 })
    );
    let mut shorter = proof;
    shorter.leaf_splits.pop();
    assert_eq!(
        verify(&lowering, &schedule, &shorter, &points),
        Err(Reject { stage: 0, round: 0 })
    );
}

/// A changed round value or final evaluation of any stage is rejected at that stage's final check,
/// and a false leaf claim on a ready claim at stage 1's (RFC-0010 'Sumcheck messages').
#[test]
fn a_changed_message_is_rejected_at_its_stage() {
    let (lowering, schedule, witness) = setup("tiny-2-minimal");
    let points = leaf_points(&schedule, 0xBAD5);
    let (proof, _) = prove(&lowering, &schedule, &witness, &points);
    for (i, stage) in proof.stages.iter().enumerate() {
        let k = stage.stage;
        let end = Reject {
            stage: k,
            round: schedule.stages[usize::from(k)].rounds,
        };
        let mut bad = proof.clone();
        bad.stages[i].sumcheck.rounds[0][0] += F::ONE;
        assert_eq!(
            verify(&lowering, &schedule, &bad, &points),
            Err(end),
            "stage {k}'s first round"
        );
        let mut bad = proof.clone();
        let last = bad.stages[i].sumcheck.final_evals.len() - 1;
        bad.stages[i].sumcheck.final_evals[last] += F::ONE;
        assert_eq!(
            verify(&lowering, &schedule, &bad, &points),
            Err(end),
            "stage {k}'s last evaluation"
        );
        let mut bad = proof.clone();
        bad.stages[i].sumcheck.rounds[0].push(F::ONE);
        assert_eq!(
            verify(&lowering, &schedule, &bad, &points),
            Err(Reject { stage: k, round: 0 }),
            "stage {k}'s first round of the wrong length"
        );
    }
    // A false claim of the leaf stage on a ready claim: stage 1's sumcheck rejects it.
    let ready = schedule.stages[0]
        .claims
        .iter()
        .position(|c| !c.split && c.parts.iter().any(|p| matches!(p, Part::Instance { .. })))
        .expect("a ready virtual claim");
    let mut bad = proof;
    bad.leaf[ready][0] += F::ONE;
    let stage1 = Reject {
        stage: 1,
        round: schedule.stages[1].rounds,
    };
    // The leaf's values are bound before stage 0's splits, so the prover's stages were proven
    // for the honest values: only the stage-1 check can see the change.
    let verdict = verify(&lowering, &schedule, &bad, &points);
    assert_eq!(verdict, Err(stage1));
}

// ----- Templates the Llama lowering does not emit ---------------------------------------------

/// A tensor of `shape` and `kind` whose declared range holds every value the templates reach.
fn decl(shape: &[u32], kind: NodeKind) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType::I32,
        range: IntRange {
            lo: -(1 << 30),
            hi: 1 << 30,
        },
        kind,
    }
}

const ADVICE: NodeKind = NodeKind::Committed {
    commitment: CommitmentClass::Advice,
};

fn one(b: &mut GraphBuilder, op: Op, inputs: &[TensorId], output: TensorDecl) -> TensorId {
    b.node(op, inputs, vec![output]).expect("a valid node")[0]
}

/// A requantization by `2` into `[lo, hi]`: a layer's last node, its checkpoint.
fn requant(b: &mut GraphBuilder, key: TensorId, shape: &[u32], bound: i64) -> TensorId {
    let shift = b.shift_classes(vec![1]).expect("shift classes");
    let op = Op::Requant {
        shift,
        lo: -bound,
        hi: bound,
    };
    let mut output = decl(shape, ADVICE);
    output.range = IntRange {
        lo: -i128::from(bound),
        hi: i128::from(bound),
    };
    one(b, op, &[key], output)
}

/// A lowering of `graph` with `x` its one input and `checkpoints` its layers' boundaries.
fn lowering_of(graph: Graph, x: TensorId, checkpoints: Vec<TensorId>) -> Lowering {
    let logits = *checkpoints.last().expect("checkpoints");
    Lowering {
        graph,
        tables: Vec::new(),
        tokens: x,
        positions: x,
        one_hot: x,
        logits,
        checkpoints,
        sites: Vec::new(),
        classes: Vec::new(),
        multipliers: Vec::new(),
    }
}

/// Proves and verifies every stage of a template on random inputs in `[−100, 100]`, and checks
/// the pool against the witness.
fn prove_template(lowering: &Lowering, inputs: &[TensorId], seed: u64) -> Proof {
    let schedule = derive_schedule(lowering).expect("a schedule");
    let mut state = seed;
    let inputs = inputs
        .iter()
        .map(|&t| {
            let spec = &lowering.graph.tensors[t.0 as usize];
            let len: usize = spec.shape.iter().map(|&d| d as usize).product();
            let span = u64::try_from(spec.range.hi - spec.range.lo + 1).expect("small");
            let values: SecretBuf<i64> = (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let offset = i64::try_from((state >> 33) % span).expect("small");
                    i64::try_from(spec.range.lo).expect("small") + offset
                })
                .collect();
            (t, values)
        })
        .collect();
    let witness = Engine::for_graph(&lowering.graph, &[])
        .expect("an engine")
        .evaluate(inputs)
        .expect("a witness");
    let points = leaf_points(&schedule, seed);
    let (proof, prover_pool) = prove(lowering, &schedule, &witness, &points);
    let pool = verify(lowering, &schedule, &proof, &points).expect("the template verifies");
    let prover_pool: Vec<F> = prover_pool
        .into_iter()
        .map(|v| v.expect("a value"))
        .collect();
    assert_eq!(pool, prover_pool);
    let coords = coords_after(&schedule, &proof, &points);
    assert_eq!(
        pool,
        pool_from_witness(&coords, lowering, &schedule, &witness)
    );
    proof
}

/// A relation's output scaled by 3 in a stacked claim, `3·c²` times `c`: stage 1 splits the
/// claim on the scaled square, one value of coefficient 3, which stage 2 proves; a changed value
/// is rejected at stage 1.
#[test]
fn a_scaled_relation_output_splits_in_a_reduction_stage() {
    let mut b = GraphBuilder::new();
    let mut input = decl(&[4, 8], ADVICE);
    input.range = IntRange { lo: -100, hi: 100 };
    let x = b.input(input).expect("an input");
    let unit = b.constant(vec![1]).expect("a constant");
    let mut c = one(
        &mut b,
        Op::ConstMul { c: unit },
        &[x],
        decl(&[4, 8], ADVICE),
    );
    let mut checkpoints = vec![c];
    for _ in 0..2 {
        let mul = Op::Mul {
            broadcast: tessera_ir::Broadcast::Trailing,
        };
        let square = one(
            &mut b,
            mul.clone(),
            &[c, c],
            decl(&[4, 8], NodeKind::Virtual),
        );
        let three = b.constant(vec![3]).expect("a constant");
        let scaled = one(
            &mut b,
            Op::ConstMul { c: three },
            &[square],
            decl(&[4, 8], NodeKind::Virtual),
        );
        let cube = one(&mut b, mul, &[scaled, c], decl(&[4, 8], NodeKind::Virtual));
        c = requant(&mut b, cube, &[4, 8], 100);
        checkpoints.push(c);
    }
    let lowering = lowering_of(b.finish(), x, checkpoints);
    let schedule = derive_schedule(&lowering).expect("a schedule");
    assert!(schedule.stages[1].claims.iter().any(|claim| claim.split));
    let proof = prove_template(&lowering, &[x], 0x0113_3C2E);
    assert_eq!(proof.stages[0].splits.len(), 1, "the scaled square's value");
    let points = leaf_points(&schedule, 0x0113_3C2E);
    let mut bad = proof;
    bad.stages[0].splits[0] += F::ONE;
    assert_eq!(
        verify(&lowering, &schedule, &bad, &points),
        Err(Reject {
            stage: 1,
            round: schedule.stages[1].rounds
        })
    );
}

/// Every linear map in a template of its own, as `tessera-compiler`'s schedule tests build it:
/// two transposes, `Expand` and `Sum`, a constant of each layer's own, two committed groups at two
/// points (a split of the leaf stage), and a committed output in each layer, whose stacked
/// equality zero-check reads each layer's constant.
#[test]
fn every_linear_map_proves_in_a_template_of_its_own() {
    let mut b = GraphBuilder::new();
    let mut input = decl(&[5, 3, 4], ADVICE);
    input.range = IntRange { lo: -100, hi: 100 };
    let x = b.input(input).expect("an input");
    let two = b.constant(vec![2]).expect("a constant");
    let virt = |shape: &[u32]| decl(shape, NodeKind::Virtual);
    let mut c = one(
        &mut b,
        Op::ConstMul { c: two },
        &[x],
        decl(&[5, 3, 4], ADVICE),
    );
    let mut checkpoints = vec![c];
    for layer in 0..2 {
        let turned = one(
            &mut b,
            Op::Transpose {
                perm: vec![2, 0, 1],
            },
            &[c],
            virt(&[4, 5, 3]),
        );
        let back = one(
            &mut b,
            Op::Transpose {
                perm: vec![1, 2, 0],
            },
            &[turned],
            virt(&[5, 3, 4]),
        );
        let wide = one(
            &mut b,
            Op::Expand { axis: 0, len: 3 },
            &[c],
            virt(&[3, 5, 3, 4]),
        );
        let thrice = one(&mut b, Op::Sum { axis: 0 }, &[wide], virt(&[5, 3, 4]));
        let k = b.constant(vec![2 + 3 * layer]).expect("a constant");
        let scaled = one(&mut b, Op::ConstMul { c: k }, &[thrice], virt(&[5, 3, 4]));
        let difference = one(&mut b, Op::Sub, &[back, scaled], virt(&[5, 3, 4]));
        // Over the checkpoint's axis of 4: its padded positions are `x`'s, a bare committed
        // input's, so a sum over its axis of 5 is refused (RFC-0072 'Summing relations').
        let mean = one(&mut b, Op::Sum { axis: 2 }, &[c], virt(&[5, 3]));
        let spread = one(
            &mut b,
            Op::Expand { axis: 2, len: 4 },
            &[mean],
            virt(&[5, 3, 4]),
        );
        let key = one(&mut b, Op::Add, &[difference, spread], virt(&[5, 3, 4]));
        one(
            &mut b,
            Op::ConstMul { c: k },
            &[c],
            decl(&[5, 3, 4], ADVICE),
        );
        let split = one(&mut b, Op::Reshape, &[key], virt(&[5, 3, 2, 2]));
        let merged = one(&mut b, Op::Reshape, &[split], virt(&[5, 3, 4]));
        c = requant(&mut b, merged, &[5, 3, 4], 1000);
        checkpoints.push(c);
    }
    let lowering = lowering_of(b.finish(), x, checkpoints);
    let proof = prove_template(&lowering, &[x], 0x0113_E7E2);
    assert!(!proof.leaf_splits.is_empty(), "the key's two groups");
}

/// A product by each layer's own channel multipliers, `Mul`'s `Leading` broadcast stacked over the
/// layers: the stage lays the multipliers out over the rows, and their claim is at their channels'
/// and layers' coordinates (RFC-0025's affine key).
#[test]
fn a_stacked_channel_product_proves() {
    let mut b = GraphBuilder::new();
    let mut input = decl(&[5, 8], ADVICE);
    input.range = IntRange { lo: -100, hi: 100 };
    let x = b.input(input).expect("an input");
    let unit = b.constant(vec![1]).expect("a constant");
    let mut c = one(
        &mut b,
        Op::ConstMul { c: unit },
        &[x],
        decl(&[5, 8], ADVICE),
    );
    let mut checkpoints = vec![c];
    let mut inputs = vec![x];
    for _ in 0..3 {
        let mut multipliers = decl(
            &[8],
            NodeKind::Committed {
                commitment: CommitmentClass::Weight,
            },
        );
        multipliers.range = IntRange { lo: 1, hi: 100 };
        let m = b.input(multipliers).expect("the multipliers");
        inputs.push(m);
        let mul = Op::Mul {
            broadcast: tessera_ir::Broadcast::Leading,
        };
        let key = one(&mut b, mul, &[c, m], decl(&[5, 8], NodeKind::Virtual));
        c = requant(&mut b, key, &[5, 8], 100);
        checkpoints.push(c);
    }
    let lowering = lowering_of(b.finish(), x, checkpoints);
    let schedule = derive_schedule(&lowering).expect("a schedule");
    assert!(
        schedule.stages[1]
            .instances
            .iter()
            .any(|i| i.relation == Relation::Mul(tessera_ir::Broadcast::Leading))
    );
    prove_template(&lowering, &inputs, 0x0113_A1F1);
}

// ----- Graphs without layers (RFC-0072) ---------------------------------------------------------

/// Committed `a`, `b` and `c` of `[3, 8]` and a weight `W`: `s = a·b + W·c` requantized, its sum
/// of squares over its channels requantized, and `3a` committed. No checkpoint and no logits. The
/// graph and its inputs `a`, `b`, `c` and `W`.
fn products() -> (Graph, [TensorId; 4]) {
    let mut g = GraphBuilder::new();
    let mut input = decl(&[3, 8], ADVICE);
    // Small enough that the sum of squares stays inside its declared range.
    input.range = IntRange { lo: -3, hi: 3 };
    let a = g.input(input.clone()).expect("a");
    let b = g.input(input.clone()).expect("b");
    let c = g.input(input).expect("c");
    let weight = WeightRef {
        ty: WeightType::Q,
        layer: 0,
    };
    let w = g.weight(weight, 8, 8).expect("a weight");
    let mul = Op::Mul {
        broadcast: tessera_ir::Broadcast::Trailing,
    };
    let virt = |shape: &[u32]| decl(shape, NodeKind::Virtual);
    let m = one(&mut g, mul, &[a, b], virt(&[3, 8]));
    let p = one(&mut g, Op::MatMulWeight { weight }, &[c, w], virt(&[3, 8]));
    let s = one(&mut g, Op::Add, &[m, p], virt(&[3, 8]));
    let q = one(&mut g, Op::SumSquares { axis: 1 }, &[s], virt(&[3]));
    requant(&mut g, q, &[3], 100);
    requant(&mut g, s, &[3, 8], 100);
    let three = g.constant(vec![3]).expect("a constant");
    one(
        &mut g,
        Op::ConstMul { c: three },
        &[a],
        decl(&[3, 8], ADVICE),
    );
    (g.finish(), [a, b, c, w])
}

/// Random values in each input's declared range.
fn random_inputs(graph: &Graph, inputs: &[TensorId], seed: u64) -> Vec<(TensorId, SecretBuf<i64>)> {
    let mut state = seed;
    inputs
        .iter()
        .map(|&t| {
            let spec = &graph.tensors[t.0 as usize];
            let len: usize = spec.shape.iter().map(|&d| d as usize).product();
            let span = u64::try_from(spec.range.hi - spec.range.lo + 1).expect("small");
            let values: SecretBuf<i64> = (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let offset = i64::try_from((state >> 33) % span).expect("small");
                    i64::try_from(spec.range.lo).expect("small") + offset
                })
                .collect();
            (t, values)
        })
        .collect()
}

/// A witness whose tensor `tensor` holds `values`, every other the honest witness's: a prover that
/// commits an input other than the one its virtual tensors were computed from.
struct Tampered<'a> {
    witness: &'a Witness,
    tensor: TensorId,
    values: Vec<i64>,
}

impl WitnessSource for Tampered<'_> {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        if tensor == self.tensor {
            return Some(&self.values);
        }
        self.witness.values(tensor)
    }
}

/// RFC-0072 'Graphs without layers': with no checkpoint, `L = λ = 0`, stage 1 squeezes no layer
/// coordinate, and every relation runs on tensors outside the layers; with no logits the leaf
/// stage leaves claims on its sites alone. The schedule proves through a prover of the graph alone
/// and the in-test verifier accepts it, its pool the committed tensors' extensions. A committed
/// input entry changed under the honest values fails stage 1's final check.
#[test]
fn a_graph_without_layers_proves_and_verifies() {
    let (graph, inputs) = products();
    let schedule = derive_graph_schedule(&graph, &[], &[], None).expect("a schedule");
    assert_eq!((schedule.layers.count, schedule.layers.vars), (0, 0));
    assert_eq!(schedule.stage_count(), 3);
    assert_eq!(
        schedule.stage_count(),
        tessera_compiler::stage_count(&graph) as usize,
        "the report's count"
    );
    let witness = Engine::for_graph(&graph, &[])
        .expect("an engine")
        .evaluate(random_inputs(&graph, &inputs, 0x0526_0001))
        .expect("a witness");
    let points = leaf_points(&schedule, 0x0526_0001);
    let (proof, prover_pool) = prove_graph(&graph, &[], &schedule, &witness, &points);
    assert_eq!(proof.stages.len() + 1, schedule.stage_count());
    let pool = verify_graph(&graph, &[], &schedule, &proof, &points).expect("it verifies");
    let prover_pool: Vec<F> = prover_pool
        .into_iter()
        .map(|v| v.expect("a value"))
        .collect();
    assert_eq!(pool, prover_pool);
    let coords = coords_after(&schedule, &proof, &points);
    assert_eq!(pool, pool_of_graph(&coords, &graph, &schedule, &witness));
    // The prover of a lowering proves the same messages.
    let lowering = Lowering {
        graph: graph.clone(),
        tables: Vec::new(),
        tokens: inputs[0],
        positions: inputs[0],
        one_hot: inputs[0],
        logits: inputs[0],
        checkpoints: Vec::new(),
        sites: Vec::new(),
        classes: Vec::new(),
        multipliers: Vec::new(),
    };
    let (same, _) = prove(&lowering, &schedule, &witness, &points);
    assert_eq!(same.stages, proof.stages);
    // `b`, which only the products read, with one entry changed.
    let b = inputs[1];
    let mut values = witness.tensor(b).expect("b").to_vec();
    values[5] += 1;
    let tampered = Tampered {
        witness: &witness,
        tensor: b,
        values,
    };
    let (bad, _) = prove_graph(&graph, &[], &schedule, &tampered, &points);
    assert_eq!(
        verify_graph(&graph, &[], &schedule, &bad, &points),
        Err(Reject {
            stage: 1,
            round: schedule.stages[1].rounds
        })
    );
}

/// RFC-0056: the stage driver must share the original operand and equality construction,
/// before the sumcheck engine shares its folded tables. Five row-wise squares leave 32
/// distinct claims on one norm. Zero ranges keep that synthetic circuit inside true integers.
#[test]
fn repeated_sum_squares_claims_allocate_one_stage_table_set() {
    let measure = |depth: u32| {
        let mut builder = GraphBuilder::new();
        let mut input = decl(&[2, 1 << 15], ADVICE);
        input.range = IntRange { lo: 0, hi: 0 };
        let x = builder.input(input).unwrap();
        let mut y = one(
            &mut builder,
            Op::SumSquares { axis: 1 },
            &[x],
            decl(&[2], NodeKind::Virtual),
        );
        for _ in 0..depth {
            y = one(
                &mut builder,
                Op::Mul {
                    broadcast: tessera_ir::Broadcast::Trailing,
                },
                &[y, y],
                decl(&[2], NodeKind::Virtual),
            );
        }
        requant(&mut builder, y, &[2], 1);
        let graph = builder.finish();
        let schedule = derive_graph_schedule(&graph, &[], &[], None).unwrap();
        let count = schedule
            .stages
            .iter()
            .flat_map(|s| &s.instances)
            .filter(|i| i.relation == Relation::SumSquares)
            .count();
        assert_eq!(count, 1 << depth, "claims remain separate");
        let witness = Engine::for_graph(&graph, &[])
            .unwrap()
            .evaluate(random_inputs(&graph, &[x], 0x461))
            .unwrap();
        let points = leaf_points(&schedule, 0x461);
        let mut proof = None;
        let info = allocation_counter::measure(|| {
            proof = Some(prove_graph(&graph, &[], &schedule, &witness, &points).0);
        });
        verify_graph(&graph, &[], &schedule, &proof.unwrap(), &points)
            .unwrap_or_else(|error| panic!("depth {depth}: {error:?}"));
        info
    };
    let single = measure(0);
    let repeated = measure(5);
    assert!(
        repeated.bytes_max < single.bytes_max + 256 * 1024,
        "single={single:?} repeated={repeated:?}"
    );
    assert!(
        repeated.bytes_total < single.bytes_total + 512 * 1024,
        "single={single:?} repeated={repeated:?}"
    );
}

/// RFC-0072 'Gadgets and leaf sites': the leaf layout gives `BitDecompose` and `AssertEqual` no
/// site, as the schedule gives none (`has_leaf_site`, which both call), and every other gadget,
/// `OneHot` among them, its own.
#[test]
fn the_bit_operations_take_no_leaf_site() {
    let mut g = GraphBuilder::new();
    let mut input = decl(&[4], ADVICE);
    input.range = IntRange { lo: 0, hi: 3 };
    let x = g.input(input).expect("x");
    g.node(
        Op::BitDecompose { low: 2, high: 0 },
        &[x],
        vec![decl(&[4, 2], ADVICE)],
    )
    .expect("bits");
    let r = requant(&mut g, x, &[4], 100);
    g.node(Op::AssertEqual, &[x, r], Vec::new())
        .expect("an equality");
    one(&mut g, Op::OneHot { vocab: 4 }, &[x], decl(&[4, 4], ADVICE));
    let graph = g.finish();
    let lowering = Lowering {
        graph,
        tables: Vec::new(),
        tokens: x,
        positions: x,
        one_hot: x,
        logits: x,
        checkpoints: Vec::new(),
        sites: Vec::new(),
        classes: Vec::new(),
        multipliers: Vec::new(),
    };
    let layers = LayerMap::new(&lowering).expect("no checkpoint, no layers");
    assert_eq!(layers.count, 0);
    let sites = gadget_nodes(&lowering, &layers);
    let expected: Vec<NodeRef> = lowering
        .graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::Requant { .. } | Op::OneHot { .. }))
        .map(|n| NodeRef::Node(n.id))
        .collect();
    assert_eq!(sites, expected);
}

/// RFC-0072 'Padded positions': an S-box, a lookup non-zero at padding, holds `AesSbox(0) =
/// 0x63` at its padded positions, which the engine's witness of the real positions, padded with
/// zeros, does not hold. The stage prover refuses that witness at its leaf stage, and takes a
/// [`PaddedWitness`] of it, whose S-box table holds the S-box of each real byte and `0x63` at
/// each padded position. An S-box with no padded position, and a `Gf256Mul` lookup with padded
/// positions, which maps 0 to 0, need none.
#[test]
fn a_witness_without_the_padded_positions_of_a_lookup_non_zero_there_is_refused() {
    let leaf = |kind: TableKind, shape: &[u32], padded: bool| {
        let mut g = GraphBuilder::new();
        let mut input = decl(shape, ADVICE);
        input.range = IntRange { lo: 0, hi: 255 };
        let x = g.input(input).expect("bytes");
        let output = decl(shape, NodeKind::LookupOutput { table: TableId(0) });
        let s = g
            .node(Op::TableMap { table: TableId(0) }, &[x], vec![output])
            .expect("a lookup")[0];
        let graph = g.finish();
        let generated = GeneratedTable::new(TableId(0), kind).expect("a table");
        let tables = vec![generated.spec().clone()];
        let schedule = derive_graph_schedule(&graph, &tables, &[], None).expect("a schedule");
        let witness = Engine::for_graph(&graph, &tables)
            .expect("an engine")
            .evaluate(random_inputs(&graph, &[x], 0x0526_0002))
            .expect("a witness");
        let points = leaf_points(&schedule, 0x0526_0002);
        if !padded {
            let mut prover = StageProver::for_graph(&graph, &tables, &schedule, &witness);
            return prover.leaf(&points).map(|_| ());
        }
        let source = PaddedWitness::new(&graph, &tables, &witness).expect("padded tables");
        // The S-box's table: each real byte's S-box, and the table's value at 0 elsewhere.
        if let Some(table) = source.padded(s) {
            let cube: Vec<usize> = shape
                .iter()
                .map(|&d| d.next_power_of_two() as usize)
                .collect();
            let bytes = witness.tensor(x).expect("the bytes");
            for (index, &value) in table.iter().enumerate() {
                let (row, column) = (index / cube[1], index % cube[1]);
                let real = row < shape[0] as usize && column < shape[1] as usize;
                let key = if real {
                    bytes[row * shape[1] as usize + column]
                } else {
                    0
                };
                assert_eq!(Some(value), generated.lookup(key, 0), "entry {index}");
            }
        }
        let mut prover = StageProver::for_graph(&graph, &tables, &schedule, &source);
        prover.leaf(&points).map(|_| ())
    };
    assert_eq!(
        leaf(TableKind::AesSbox, &[4, 3], false),
        Err(StageError::Internal {
            rule: "the witness holds the padded positions of every tensor a lookup non-zero at \
                   padding reaches"
        })
    );
    leaf(TableKind::AesSbox, &[4, 3], true).expect("the padded witness");
    leaf(TableKind::AesSbox, &[4, 4], false).expect("no padded position");
    leaf(TableKind::Gf256Mul { c: 2 }, &[4, 3], false).expect("a lookup that maps 0 to 0");
}

// ----- Public terms, ConstAdd and the per-channel ConstMul (RFC-0072) ---------------------------

/// The public tensors a verifier computes from the public graph inputs and constants alone
/// (RFC-0072 'Public tensors'): the graph run on the public inputs `public`, every other input 0,
/// of which only the public tensors are kept. By the closure no committed input reaches a public
/// tensor, so the zeros change none of them.
struct PublicValues {
    witness: Witness,
    public: Vec<bool>,
}

impl WitnessSource for PublicValues {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        if !self.public.get(tensor.0 as usize).copied().unwrap_or(false) {
            return None;
        }
        self.witness.values(tensor)
    }
}

fn public_values(graph: &Graph, public: &[(TensorId, Vec<i64>)]) -> PublicValues {
    let flags = public_tensors(graph);
    let inputs = graph
        .tensors
        .iter()
        .filter(|t| !graph.nodes.iter().any(|n| n.outputs.contains(&t.id)))
        .map(|t| {
            let values: SecretBuf<i64> = match public.iter().find(|(id, _)| *id == t.id) {
                Some((_, values)) => {
                    assert!(flags[t.id.0 as usize], "a public input");
                    values.iter().copied().collect()
                }
                None => {
                    let len: usize = t.shape.iter().map(|&d| d as usize).product();
                    (0..len).map(|_| 0).collect()
                }
            };
            (t.id, values)
        })
        .collect();
    let witness = Engine::for_graph(graph, &[])
        .expect("an engine")
        .evaluate(inputs)
        .expect("the public tensors");
    PublicValues {
        witness,
        public: flags,
    }
}

/// A graph's honest run: its schedule, the witness of random inputs in their declared ranges,
/// the public inputs' values, the leaf points, the proof and the prover's pool.
struct Run {
    schedule: StageSchedule,
    witness: Witness,
    public: Vec<(TensorId, Vec<i64>)>,
    points: Vec<Vec<F>>,
    proof: Proof,
    pool: Vec<F>,
}

impl Run {
    fn of(graph: &Graph, inputs: &[TensorId], seed: u64) -> Self {
        let schedule = derive_graph_schedule(graph, &[], &[], None).expect("a schedule");
        assert_eq!(
            schedule.stage_count(),
            tessera_compiler::stage_count(graph) as usize,
            "the report's count"
        );
        let values = random_inputs(graph, inputs, seed);
        let flags = public_tensors(graph);
        let public = values
            .iter()
            .filter(|(t, _)| flags[t.0 as usize])
            .map(|(t, v)| (*t, v.to_vec()))
            .collect();
        let witness = Engine::for_graph(graph, &[])
            .expect("an engine")
            .evaluate(values)
            .expect("a witness");
        let points = leaf_points(&schedule, seed);
        let (proof, pool) = prove_graph(graph, &[], &schedule, &witness, &points);
        let pool = pool.into_iter().map(|v| v.expect("a value")).collect();
        Self {
            schedule,
            witness,
            public,
            points,
            proof,
            pool,
        }
    }

    /// The verifier's verdict on `proof`, with the public inputs `public` and the shift `shift`.
    fn verify_with(
        &self,
        graph: &Graph,
        proof: &Proof,
        public: &[(TensorId, Vec<i64>)],
        shift: Shift,
    ) -> Result<Vec<F>, Reject> {
        let values = public_values(graph, public);
        let publics = Publics {
            values: &values,
            shift,
            kernel: Kernel::Closed,
        };
        verify_public(graph, &[], &self.schedule, proof, &self.points, publics)
    }

    /// The honest verifier accepts the honest proof, its pool the prover's and the committed
    /// tensors' extensions.
    fn accepted(&self, graph: &Graph) {
        let pool = self
            .verify_with(graph, &self.proof, &self.public, Shift::Extension)
            .expect("the honest proof verifies");
        assert_eq!(pool, self.pool, "the prover's pool");
        let coords = coords_after(&self.schedule, &self.proof, &self.points);
        assert_eq!(
            pool,
            pool_of_graph(&coords, graph, &self.schedule, &self.witness),
            "the pool is the committed tensors' extensions"
        );
    }

    /// The proof of a prover whose committed input `tensor` has entry `entry` changed under the
    /// honest values.
    fn tampered(&self, graph: &Graph, tensor: TensorId, entry: usize) -> Proof {
        let mut values = self.witness.tensor(tensor).expect("the input").to_vec();
        values[entry] += 1;
        let tampered = Tampered {
            witness: &self.witness,
            tensor,
            values,
        };
        prove_graph(graph, &[], &self.schedule, &tampered, &self.points).0
    }

    /// The final check of stage `k`.
    fn end(&self, k: u16) -> Reject {
        Reject {
            stage: k,
            round: self.schedule.stages[usize::from(k)].rounds,
        }
    }
}

/// The extension of `ψ·c` at `r` by enumeration over the padded cube of `shape`: the constant at
/// each real position, one value or one per channel of the last axis, 0 at every padded one,
/// weighted by `eq(r, x)` (RFC-0072 'Padded positions', 'Linear operations that are not
/// coordinate maps').
fn shift_by_enumeration(shape: &[u32], constant: &[i64], r: &[F]) -> F {
    let vars: Vec<u32> = shape.iter().map(|&d| axis_vars(d)).collect();
    let n: u32 = vars.iter().sum();
    assert_eq!(r.len(), n as usize);
    let mut sum = F::ZERO;
    for x in 0..1u64 << n {
        let (mut rest, mut real, mut channel) = (x, true, 0);
        for (axis, &len) in shape.iter().enumerate().rev() {
            let k = rest & ((1 << vars[axis]) - 1);
            rest >>= vars[axis];
            real &= k < u64::from(len);
            if axis + 1 == shape.len() {
                channel = k as usize;
            }
        }
        if real {
            let c = if constant.len() == 1 {
                constant[0]
            } else {
                constant[channel]
            };
            sum += F::from(Goldilocks::from_i64(c)) * eq_at(r, x);
        }
    }
    sum
}

/// Random points of `n` coordinates.
fn random_points(n: usize, count: usize, seed: u64) -> Vec<Vec<F>> {
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        Goldilocks::from_u64_reduced(rng >> 11)
    };
    (0..count)
        .map(|_| (0..n).map(|_| F::new(next(), next())).collect())
        .collect()
}

const PUBLIC: NodeKind = NodeKind::Public;

/// An input of `shape` and `kind` with values in `[−3, 3]`.
fn small(g: &mut GraphBuilder, shape: &[u32], kind: NodeKind) -> TensorId {
    let mut input = decl(shape, kind);
    input.range = IntRange { lo: -3, hi: 3 };
    g.input(input).expect("an input")
}

fn product(broadcast: tessera_ir::Broadcast) -> Op {
    Op::Mul { broadcast }
}

/// Where a test's `ConstAdd` sits, and so which check reads its shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shifted {
    /// `requant(a·b + c)`: the leaf claim on the sum becomes `a·b`'s at `v − ĉ(r)`, which stage
    /// 1's product proves.
    Virtual,
    /// `requant(x + c)`, `x` committed: the leaf claim is ready on `x`, whose pool claim is
    /// `v − ĉ(r)`, which the opening checks.
    Pooled,
    /// `y = x + c` committed: stage 1's zero-check `Ỹ − X̃ − ĉ`, `ĉ` at its final point.
    Committed,
}

/// The shapes a test's `ConstAdd` runs on: RFC-0072's last axis of 3, and the degenerate ones a
/// selector can get wrong, an axis of length 1, a single axis, three axes, and lengths that are
/// powers of two, where `c̃(r_c)·ψ̃(r)` is `ĉ`.
const SHIFT_SHAPES: [&[u32]; 10] = [
    &[3],
    &[3, 1],
    &[1, 3],
    &[5, 3],
    &[2, 3, 6],
    &[7, 5],
    &[4],
    &[2, 1, 3],
    &[4, 4],
    &[1],
];

/// The constant of a test's `ConstAdd` on a last axis of `channels`: one value, or one per
/// channel, which a last axis of 1 does not have ('Scalar or per channel').
fn shift_constant(channels: u32, per_channel: bool) -> Option<Vec<i64>> {
    match (per_channel, channels) {
        (false, _) => Some(vec![9]),
        (true, 1) => None,
        (true, _) => Some((0..i64::from(channels)).map(|i| 3 * i - 7).collect()),
    }
}

/// RFC-0072 'Linear operations that are not coordinate maps' (`ConstAdd`), RFC-0037 'Claims and
/// atoms', 'Splits'; RFC-0072 'Testing Strategy' 'Relations': a `ConstAdd` by a scalar and by a
/// constant per channel, virtual and committed ([`Shifted`] says where each test's sits), proves
/// and verifies on every shape of [`SHIFT_SHAPES`], the last axis of 3 among them.
///
/// The verifier's shift is the extension of `ψ·c` enumerated over the padded cube. A verifier
/// taking `c̃(r_c)·ψ̃(r)` in its place, or no shift, rejects the honest proof at the check that
/// reads it, stage 1's final check, or else the opening, whose pool value it gets wrong; `c̃·ψ̃`
/// is `ĉ`, and accepts, for a scalar and for a last axis whose length is a power of two. A changed
/// committed entry is rejected at stage 1.
#[test]
fn a_const_add_proves_and_verifies() {
    for shape in SHIFT_SHAPES {
        let channels = *shape.last().expect("an axis");
        for per_channel in [false, true] {
            let Some(values) = shift_constant(channels, per_channel) else {
                continue;
            };
            // `c̃(r_c)·ψ̃(r)` differs from `ĉ` off the cube exactly then.
            let differs = per_channel && !channels.is_power_of_two();
            for mode in [Shifted::Virtual, Shifted::Pooled, Shifted::Committed] {
                const_add_case(shape, &values, mode, differs);
            }
        }
    }
}

/// One case of [`a_const_add_proves_and_verifies`].
fn const_add_case(shape: &[u32], values: &[i64], mode: Shifted, differs: bool) {
    let name = format!("{shape:?}, constant {values:?}, {mode:?}");
    let mut g = GraphBuilder::new();
    let c = g.constant(values.to_vec()).expect("a constant");
    let virt = decl(shape, NodeKind::Virtual);
    // The committed inputs, and the operands of the product, if any.
    let (inputs, operands) = match mode {
        Shifted::Virtual => {
            let a = small(&mut g, shape, ADVICE);
            let b = small(&mut g, shape, ADVICE);
            let mul = product(tessera_ir::Broadcast::Trailing);
            let m = one(&mut g, mul, &[a, b], virt.clone());
            let y = one(&mut g, Op::ConstAdd { c }, &[m], virt);
            requant(&mut g, y, shape, 100);
            (vec![a, b], Some((a, b)))
        }
        Shifted::Pooled => {
            let x = small(&mut g, shape, ADVICE);
            let y = one(&mut g, Op::ConstAdd { c }, &[x], virt);
            requant(&mut g, y, shape, 100);
            (vec![x], None)
        }
        Shifted::Committed => {
            let x = small(&mut g, shape, ADVICE);
            one(&mut g, Op::ConstAdd { c }, &[x], decl(shape, ADVICE));
            (vec![x], None)
        }
    };
    let graph = g.finish();
    let node = graph
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::ConstAdd { .. }))
        .expect("the ConstAdd");
    let (node, output) = (node.id, &graph.tensors[node.outputs[0].0 as usize].shape);
    let run = Run::of(&graph, &inputs, 0x0526_0A00 + u64::from(values.len() > 1));
    // The verifier's shift is the extension of `ψ·c`; `c̃·ψ̃` is not, when it differs.
    let public = PublicTensors::of(&graph);
    let vars = output.iter().map(|&d| axis_vars(d) as usize).sum();
    for r in random_points(vars, 4, 0x0526_0A10) {
        let shift = public_value(&graph, &public, PublicSource::Shift(node), &NoPublic, &r)
            .expect("a shift");
        assert_eq!(shift, shift_by_enumeration(output, values, &r), "{name}");
        let product = channels_times_selector(&graph, node, &r);
        assert_eq!(product != shift, differs, "{name}: c̃·ψ̃");
    }
    run.accepted(&graph);
    // A wrong verifier's verdict: stage 1's final check, or a pool claim the opening rejects,
    // since it is not the committed tensors' extension.
    let wrong = |shift| match run.verify_with(&graph, &run.proof, &run.public, shift) {
        Ok(pool) => {
            let coords = coords_after(&run.schedule, &run.proof, &run.points);
            let opened = pool_of_graph(&coords, &graph, &run.schedule, &run.witness);
            (pool != opened).then_some("the opening")
        }
        Err(reject) => (reject == run.end(1)).then_some("stage 1"),
    };
    let expected = if mode == Shifted::Pooled {
        "the opening"
    } else {
        "stage 1"
    };
    assert_eq!(wrong(Shift::Omitted), Some(expected), "{name}: no shift");
    let product = wrong(Shift::ChannelsTimesSelector);
    assert_eq!(product, differs.then_some(expected), "{name}: c̃(r_c)·ψ̃(r)");
    if mode == Shifted::Pooled {
        return;
    }
    // A committed entry changed under the honest values: the zero-check's `X`, or one of the
    // product's operands where the other is not 0.
    let (changed, entry) = match operands {
        None => (inputs[0], 0),
        Some((a, b)) => {
            let at = |t: TensorId| run.witness.tensor(t).expect("an operand").to_vec();
            let (va, vb) = (at(a), at(b));
            match vb.iter().position(|&v| v != 0) {
                Some(i) => (a, i),
                None => (
                    b,
                    va.iter().position(|&v| v != 0).expect("a non-zero entry"),
                ),
            }
        }
    };
    let bad = run.tampered(&graph, changed, entry);
    assert_eq!(
        run.verify_with(&graph, &bad, &run.public, Shift::Extension),
        Err(run.end(1)),
        "{name}: a changed input entry"
    );
}

/// RFC-0072 'Public tensors' 'As an operand', RFC-0010 'Relations': a `Mul` with a public operand
/// reads it as `Public`, so the instance sends one evaluation, its other operand's, and the
/// verifier evaluates the public one at its own coordinates of the instance's point. A public row
/// scale (`Trailing`), channel scale (`Leading`) and first operand each prove and verify, and so
/// does a committed product's zero-check with a public channel scale. A public operand changed on
/// the verifier's side, and a changed committed entry, are rejected at stage 1's final check.
#[test]
fn a_mul_with_a_public_operand_proves_and_verifies() {
    use tessera_ir::Broadcast::{Leading, Trailing};
    // The operands' shapes and kinds, the broadcast, and whether the product is committed.
    let case = |a: &'static [u32], a_kind, b: &'static [u32], b_kind, broadcast, committed| {
        (a, a_kind, b, b_kind, broadcast, committed)
    };
    let cases = [
        (
            "a public row scale",
            case(&[5, 3], ADVICE, &[5], PUBLIC, Trailing, false),
        ),
        (
            "a public channel scale",
            case(&[5, 3], ADVICE, &[3], PUBLIC, Leading, false),
        ),
        (
            "a public first operand",
            case(&[5, 3], PUBLIC, &[5], ADVICE, Trailing, false),
        ),
        (
            "a committed product",
            case(&[5, 3], ADVICE, &[3], PUBLIC, Leading, true),
        ),
    ];
    for (name, (a_shape, a_kind, b_shape, b_kind, broadcast, committed)) in cases {
        let mut g = GraphBuilder::new();
        let a = small(&mut g, a_shape, a_kind);
        let b = small(&mut g, b_shape, b_kind);
        let kind = if committed { ADVICE } else { NodeKind::Virtual };
        let y = one(&mut g, product(broadcast), &[a, b], decl(a_shape, kind));
        if !committed {
            requant(&mut g, y, a_shape, 100);
        }
        let graph = g.finish();
        let run = Run::of(&graph, &[a, b], 0x0526_0B00);
        let stage = &run.schedule.stages[1];
        let instance = &stage.instances[0];
        let sent = if committed { 2 } else { 1 };
        assert_eq!(
            instance.claims.len(),
            sent,
            "{name}: the public operand sends nothing"
        );
        assert_eq!(
            run.proof.stages[0].sumcheck.final_evals.len(),
            sent,
            "{name}"
        );
        run.accepted(&graph);
        // The verifier's public operand with one entry changed.
        let (public, committed_input) = if a_kind == PUBLIC { (a, b) } else { (b, a) };
        let mut changed = run.public.clone();
        let (_, values) = changed
            .iter_mut()
            .find(|(t, _)| *t == public)
            .expect("the public operand");
        values[1] += 1;
        assert_eq!(
            run.verify_with(&graph, &run.proof, &changed, Shift::Extension),
            Err(run.end(1)),
            "{name}: a changed public operand"
        );
        let bad = run.tampered(&graph, committed_input, 2);
        assert_eq!(
            run.verify_with(&graph, &bad, &run.public, Shift::Extension),
            Err(run.end(1)),
            "{name}: a changed committed entry"
        );
    }
}

/// RFC-0072 'ConstMul by a constant per channel', RFC-0010 'Relations': `Ỹ(r) = Σ_x eq(r, x)·
/// c̃(x_c)·X̃(x)`, degree 3, the constant a public operand broadcast over the other axes, 0 at the
/// padded channels of a last axis of 3, leaving one claim on `X`; and a committed output's
/// zero-check `Ỹ − c̃(x_c)·X̃`. Both prove and verify, and a changed input entry is rejected at
/// stage 1's final check.
#[test]
fn a_const_mul_per_channel_proves_and_verifies() {
    for committed in [false, true] {
        let mut g = GraphBuilder::new();
        let x = small(&mut g, &[5, 3], ADVICE);
        let c = g.constant(vec![3, -1, 4]).expect("a constant");
        let kind = if committed { ADVICE } else { NodeKind::Virtual };
        let y = one(&mut g, Op::ConstMul { c }, &[x], decl(&[5, 3], kind));
        if !committed {
            requant(&mut g, y, &[5, 3], 100);
        }
        let graph = g.finish();
        let run = Run::of(&graph, &[x], 0x0526_0C00);
        let relation = run.schedule.stages[1].instances[0].relation;
        let expected = if committed {
            Relation::Equality
        } else {
            Relation::ScaleChannels
        };
        assert_eq!(relation, expected);
        assert_eq!(run.proof.stages[0].sumcheck.rounds[0].len(), 3, "degree 3");
        run.accepted(&graph);
        let bad = run.tampered(&graph, x, 7);
        assert_eq!(
            run.verify_with(&graph, &bad, &run.public, Shift::Extension),
            Err(run.end(1)),
            "committed {committed}: a changed input entry"
        );
    }
}

/// Where a test's claim meets the public tensor `Expand(p)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reaching {
    /// `requant(a·b + Expand(p))`: the claim is `a·b`'s at `v − Ẽ(r)`, which stage 1 proves.
    Product,
    /// `requant(x + Expand(p))`, `x` committed: the pool claim on `x` is `v − Ẽ(r)`.
    Committed,
    /// `requant(Expand(p))`: the leaf site reads the public tensor, whose claim is one public
    /// term with no part: nothing is sent or proven for it (RFC-0045 'The split').
    Alone,
}

/// RFC-0010 'Relations' ('A public input, an offset's constants, is evaluated by the verifier and
/// moves to the claim's value'), RFC-0072 'Public tensors' 'Their claims': a claim through linear
/// maps to the public tensor `Expand(p)`, which the verifier computes from `p`, moves it to the
/// claim's value ([`Reaching`]). Each proves and verifies, and a `p` changed on the verifier's side
/// fails the check that reads it: stage 1's or the opening's. A requantization of `Expand(p)`
/// alone reads the public tensor at its leaf site: its claim has no part, and a changed `p` fails
/// the leaf stage's own check of it, `Reject(Sumcheck { stage: 0, round: 0 })`, which a real leaf
/// stage makes on the split's key identity instead.
#[test]
fn a_claim_reaching_a_public_tensor_proves_and_verifies() {
    for reaching in [Reaching::Product, Reaching::Committed, Reaching::Alone] {
        let mut g = GraphBuilder::new();
        let a = small(&mut g, &[4, 3], ADVICE);
        let b = small(&mut g, &[4, 3], ADVICE);
        let p = small(&mut g, &[3], PUBLIC);
        let wide = one(
            &mut g,
            Op::Expand { axis: 0, len: 4 },
            &[p],
            decl(&[4, 3], NodeKind::Virtual),
        );
        match reaching {
            Reaching::Product | Reaching::Committed => {
                let m = if reaching == Reaching::Product {
                    one(
                        &mut g,
                        product(tessera_ir::Broadcast::Trailing),
                        &[a, b],
                        decl(&[4, 3], NodeKind::Virtual),
                    )
                } else {
                    a
                };
                let s = one(
                    &mut g,
                    Op::Add,
                    &[m, wide],
                    decl(&[4, 3], NodeKind::Virtual),
                );
                requant(&mut g, s, &[4, 3], 100);
            }
            Reaching::Alone => {
                requant(&mut g, wide, &[4, 3], 100);
            }
        }
        let graph = g.finish();
        let run = Run::of(&graph, &[a, b, p], 0x0526_0D00);
        let leaf = &run.schedule.stages[0].claims[0];
        assert_eq!(leaf.public.len(), 1);
        assert_eq!(leaf.public[0].source, PublicSource::Tensor(wide));
        assert_eq!(
            leaf.parts.is_empty(),
            reaching == Reaching::Alone,
            "{reaching:?}: a part unless the claim is public alone"
        );
        run.accepted(&graph);
        let mut changed = run.public.clone();
        changed[0].1[2] += 1;
        assert_eq!(changed[0].0, p);
        let verdict = run.verify_with(&graph, &run.proof, &changed, Shift::Extension);
        match reaching {
            Reaching::Product => assert_eq!(verdict, Err(run.end(1)), "{reaching:?}"),
            Reaching::Committed => {
                let coords = coords_after(&run.schedule, &run.proof, &run.points);
                let opened = pool_of_graph(&coords, &graph, &run.schedule, &run.witness);
                assert_ne!(
                    verdict.expect("the stages pass"),
                    opened,
                    "the opening rejects"
                );
            }
            Reaching::Alone => assert_eq!(verdict, Err(run.end(0)), "{reaching:?}"),
        }
    }
}

// ----- Public terms' coefficients, shapes and points (the slice-1b review) -----------------------

/// A wrong verifier's verdict on an honest run: `Some(where)` when it rejects, at a stage or at the
/// opening, whose pool value it gets wrong.
fn rejected(run: &Run, graph: &Graph, shift: Shift) -> Option<&'static str> {
    match run.verify_with(graph, &run.proof, &run.public, shift) {
        Ok(pool) => {
            let coords = coords_after(&run.schedule, &run.proof, &run.points);
            let opened = pool_of_graph(&coords, graph, &run.schedule, &run.witness);
            (pool != opened).then_some("the opening")
        }
        Err(_) => Some("a stage"),
    }
}

/// RFC-0037 'Claims and atoms', RFC-0010 'Relations': a `Sum` over an axis places its input at
/// `½` with the coefficient `2^j`, which a `ConstAdd` below it passes to its shift. A sum over a
/// `ConstAdd`, by a scalar and per channel, proves and verifies on axis 0 of `[4, 3]`, axis 1 of
/// `[5, 4]` and axis 0 of `[2, 3, 6]`, each summed axis a power of two; a verifier with no shift
/// rejects.
#[test]
fn a_sum_through_a_const_add_proves_and_verifies() {
    let cases: [(&[u32], u8, &[u32]); 3] = [
        (&[4, 3], 0, &[3]),
        (&[5, 4], 1, &[5]),
        (&[2, 3, 6], 0, &[3, 6]),
    ];
    for (shape, axis, out) in cases {
        let channels = *shape.last().expect("an axis");
        for per_channel in [false, true] {
            let name = format!("Sum over axis {axis} of {shape:?}, per channel {per_channel}");
            let values = if per_channel {
                (0..i64::from(channels)).map(|i| 3 * i - 4).collect()
            } else {
                vec![9]
            };
            let mut g = GraphBuilder::new();
            let c = g.constant(values).expect("a constant");
            let a = small(&mut g, shape, ADVICE);
            let b = small(&mut g, shape, ADVICE);
            let virt = decl(shape, NodeKind::Virtual);
            let mul = product(tessera_ir::Broadcast::Trailing);
            let m = one(&mut g, mul, &[a, b], virt.clone());
            let y = one(&mut g, Op::ConstAdd { c }, &[m], virt);
            let s = one(&mut g, Op::Sum { axis }, &[y], decl(out, NodeKind::Virtual));
            requant(&mut g, s, out, 1000);
            let graph = g.finish();
            let run = Run::of(&graph, &[a, b], 0x0526_0F00);
            run.accepted(&graph);
            assert!(
                rejected(&run, &graph, Shift::Omitted).is_some(),
                "{name}: no shift"
            );
        }
    }
}

/// RFC-0072 'Public tensors' 'As an operand': the layouts of a public `Mul` operand on three axes,
/// on a last axis of 1, of a public operand of the product's own shape, and in committed products'
/// zero-checks, both operands public among them, prove and verify; the public operand changed on
/// the verifier's side is rejected at stage 1's final check.
#[test]
fn a_public_mul_operand_proves_on_every_layout() {
    use tessera_ir::Broadcast::{Leading, Trailing};
    let case = |a: &'static [u32], a_kind, b: &'static [u32], b_kind, broadcast, committed| {
        (a, a_kind, b, b_kind, broadcast, committed)
    };
    let cases = [
        (
            "a row scale on three axes",
            case(&[2, 3, 6], ADVICE, &[2, 3], PUBLIC, Trailing, false),
        ),
        (
            "a channel scale on three axes",
            case(&[2, 3, 6], ADVICE, &[6], PUBLIC, Leading, false),
        ),
        (
            "a public first operand, Leading",
            case(&[2, 3, 6], PUBLIC, &[6], ADVICE, Leading, false),
        ),
        (
            "a row scale on a last axis of 1",
            case(&[3, 1], ADVICE, &[3], PUBLIC, Trailing, false),
        ),
        (
            "a public first operand on [7, 5]",
            case(&[7, 5], PUBLIC, &[7], ADVICE, Trailing, false),
        ),
        (
            "a public operand of one shape",
            case(&[5, 3], ADVICE, &[5, 3], PUBLIC, Trailing, false),
        ),
        (
            "a committed row scale",
            case(&[2, 3, 6], ADVICE, &[2, 3], PUBLIC, Trailing, true),
        ),
        (
            "a committed public first operand",
            case(&[5, 3], PUBLIC, &[5], ADVICE, Trailing, true),
        ),
        (
            "a committed product of public operands",
            case(&[5, 3], PUBLIC, &[3], PUBLIC, Leading, true),
        ),
    ];
    for (name, (a_shape, a_kind, b_shape, b_kind, broadcast, committed)) in cases {
        let mut g = GraphBuilder::new();
        let a = small(&mut g, a_shape, a_kind);
        let b = small(&mut g, b_shape, b_kind);
        let kind = if committed { ADVICE } else { NodeKind::Virtual };
        let y = one(&mut g, product(broadcast), &[a, b], decl(a_shape, kind));
        if !committed {
            requant(&mut g, y, a_shape, 100);
        }
        let graph = g.finish();
        let run = Run::of(&graph, &[a, b], 0x0526_1000);
        run.accepted(&graph);
        let public = if a_kind == PUBLIC { a } else { b };
        let mut changed = run.public.clone();
        let (_, values) = changed
            .iter_mut()
            .find(|(t, _)| *t == public)
            .expect("the public operand");
        values[1] += 1;
        assert_eq!(
            run.verify_with(&graph, &run.proof, &changed, Shift::Extension),
            Err(run.end(1)),
            "{name}: a changed public operand"
        );
    }
}

/// RFC-0072 'A committed output', 'Public tensors' 'As an operand': committed outputs whose inputs
/// are public prove and verify, each public input read as `Public` in the zero-check: a
/// `ConstAdd` (a verifier with no shift rejecting it) and a `ConstMul` of a public tensor, by a
/// scalar and per channel, and an `Add` or a `Sub` with either or both inputs public.
#[test]
fn a_committed_output_of_public_inputs_proves_and_verifies() {
    for per_channel in [false, true] {
        let values = if per_channel { vec![7, -2, 5] } else { vec![9] };
        for add in [true, false] {
            let name = format!(
                "a committed {} of a public tensor, per channel {per_channel}",
                if add { "ConstAdd" } else { "ConstMul" }
            );
            let mut g = GraphBuilder::new();
            let c = g.constant(values.clone()).expect("a constant");
            let x = small(&mut g, &[5, 3], PUBLIC);
            let op = if add {
                Op::ConstAdd { c }
            } else {
                Op::ConstMul { c }
            };
            one(&mut g, op, &[x], decl(&[5, 3], ADVICE));
            let graph = g.finish();
            let run = Run::of(&graph, &[x], 0x0526_1100);
            assert_eq!(
                run.proof.stages[0].sumcheck.final_evals.len(),
                1,
                "{name}: Y's alone"
            );
            run.accepted(&graph);
            if add {
                assert!(
                    rejected(&run, &graph, Shift::Omitted).is_some(),
                    "{name}: no shift"
                );
            }
        }
    }
    for (name, op, a_kind, b_kind) in [
        ("an Add, its first input public", Op::Add, PUBLIC, ADVICE),
        ("a Sub, its second input public", Op::Sub, ADVICE, PUBLIC),
        ("a Sub, its first input public", Op::Sub, PUBLIC, ADVICE),
        ("a Sub of public inputs", Op::Sub, PUBLIC, PUBLIC),
    ] {
        let mut g = GraphBuilder::new();
        let a = small(&mut g, &[5, 3], a_kind);
        let b = small(&mut g, &[5, 3], b_kind);
        one(&mut g, op, &[a, b], decl(&[5, 3], ADVICE));
        let graph = g.finish();
        let run = Run::of(&graph, &[a, b], 0x0526_1300);
        run.accepted(&graph);
        let sent = 1 + usize::from(a_kind != PUBLIC) + usize::from(b_kind != PUBLIC);
        assert_eq!(
            run.proof.stages[0].sumcheck.final_evals.len(),
            sent,
            "{name}"
        );
    }
}

/// RFC-0072 'ConstMul by a constant per channel': virtual and committed, on three axes, an axis of
/// length 1, a last axis of 5 and one of 4, each proves and verifies, and a changed input entry, at
/// a channel whose constant is not 0, is rejected at stage 1's final check.
#[test]
fn a_const_mul_per_channel_proves_on_every_shape() {
    let shapes: [&[u32]; 4] = [&[2, 3, 6], &[1, 3], &[7, 5], &[4, 4]];
    for shape in shapes {
        for committed in [false, true] {
            let name = format!("{shape:?}, committed {committed}");
            let channels = *shape.last().expect("an axis");
            let mut g = GraphBuilder::new();
            let x = small(&mut g, shape, ADVICE);
            let c = g
                .constant((0..i64::from(channels)).map(|i| 2 * i - 3).collect())
                .expect("a constant");
            let kind = if committed { ADVICE } else { NodeKind::Virtual };
            let y = one(&mut g, Op::ConstMul { c }, &[x], decl(shape, kind));
            if !committed {
                requant(&mut g, y, shape, 100);
            }
            let graph = g.finish();
            let run = Run::of(&graph, &[x], 0x0526_1400);
            run.accepted(&graph);
            // Entry 1 is channel 1, whose constant is −1.
            let bad = run.tampered(&graph, x, 1);
            assert_eq!(
                run.verify_with(&graph, &bad, &run.public, Shift::Extension),
                Err(run.end(1)),
                "{name}: a changed input entry"
            );
        }
    }
}

/// RFC-0037 'Splits': a split claim with a public term, `requant(a·b + Expand(p) − x + Expand(p))`
/// with `x` committed, sends `a·b`'s and `x`'s values, which must give the claim's value less the
/// term, two products of one source at one point. It proves and verifies, and a `p` changed on the
/// verifier's side fails the leaf stage's split check.
#[test]
fn a_split_claim_with_a_public_term_proves_and_verifies() {
    let mut g = GraphBuilder::new();
    let a = small(&mut g, &[4, 3], ADVICE);
    let b = small(&mut g, &[4, 3], ADVICE);
    let x = small(&mut g, &[4, 3], ADVICE);
    let p = small(&mut g, &[3], PUBLIC);
    let virt = decl(&[4, 3], NodeKind::Virtual);
    let wide = one(&mut g, Op::Expand { axis: 0, len: 4 }, &[p], virt.clone());
    let mul = product(tessera_ir::Broadcast::Trailing);
    let m = one(&mut g, mul, &[a, b], virt.clone());
    let s = one(&mut g, Op::Add, &[m, wide], virt.clone());
    let t = one(&mut g, Op::Sub, &[s, x], virt.clone());
    let u = one(&mut g, Op::Add, &[t, wide], virt);
    requant(&mut g, u, &[4, 3], 100);
    let graph = g.finish();
    let run = Run::of(&graph, &[a, b, x, p], 0x0526_1500);
    let claim = &run.schedule.stages[0].claims[0];
    assert!(claim.split, "split");
    assert_eq!(claim.public.len(), 1);
    assert_eq!(claim.public[0].coefficient.len(), 2, "two products");
    run.accepted(&graph);
    let mut changed = run.public.clone();
    changed[0].1[0] += 1;
    assert_eq!(
        run.verify_with(&graph, &run.proof, &changed, Shift::Extension),
        Err(Reject { stage: 0, round: 0 }),
        "a changed public input"
    );
}

/// RFC-0072 'Public tensors' 'Once per point': terms of one public tensor merge only at one point.
/// In `requant(Transpose(a·b + P) + P)`, `P` public, the claim reads `P` at its point with the axes
/// swapped and at its point: two terms, which prove and verify.
#[test]
fn one_public_tensor_at_two_points_is_two_terms() {
    let mut g = GraphBuilder::new();
    let a = small(&mut g, &[3, 3], ADVICE);
    let b = small(&mut g, &[3, 3], ADVICE);
    let p = small(&mut g, &[3, 3], PUBLIC);
    let virt = decl(&[3, 3], NodeKind::Virtual);
    let mul = product(tessera_ir::Broadcast::Trailing);
    let m = one(&mut g, mul, &[a, b], virt.clone());
    let s = one(&mut g, Op::Add, &[m, p], virt.clone());
    let t = one(
        &mut g,
        Op::Transpose { perm: vec![1, 0] },
        &[s],
        virt.clone(),
    );
    let u = one(&mut g, Op::Add, &[t, p], virt);
    requant(&mut g, u, &[3, 3], 100);
    let graph = g.finish();
    let run = Run::of(&graph, &[a, b, p], 0x0526_1600);
    let public = &run.schedule.stages[0].claims[0].public;
    assert_eq!(public.len(), 2, "two points");
    assert_ne!(public[0].point, public[1].point);
    run.accepted(&graph);
}

/// RFC-0037 'Claims and atoms': a coefficient reaches the public terms below it. In
/// `requant(3·(a·b − Expand(p)) + 3·(x + c))`, split, the expansion's term has the coefficient
/// `−3` and the shift `3`; it proves and verifies.
#[test]
fn scaled_public_terms_of_a_split_claim_prove_and_verify() {
    let mut g = GraphBuilder::new();
    let a = small(&mut g, &[4, 3], ADVICE);
    let b = small(&mut g, &[4, 3], ADVICE);
    let x = small(&mut g, &[4, 3], ADVICE);
    let p = small(&mut g, &[3], PUBLIC);
    let three = g.constant(vec![3]).expect("a constant");
    let c = g.constant(vec![7, -2, 5]).expect("a constant");
    let virt = decl(&[4, 3], NodeKind::Virtual);
    let wide = one(&mut g, Op::Expand { axis: 0, len: 4 }, &[p], virt.clone());
    let mul = product(tessera_ir::Broadcast::Trailing);
    let m = one(&mut g, mul, &[a, b], virt.clone());
    let d = one(&mut g, Op::Sub, &[m, wide], virt.clone());
    let e = one(&mut g, Op::ConstMul { c: three }, &[d], virt.clone());
    let y = one(&mut g, Op::ConstAdd { c }, &[x], virt.clone());
    let f = one(&mut g, Op::ConstMul { c: three }, &[y], virt.clone());
    let s = one(&mut g, Op::Add, &[e, f], virt);
    requant(&mut g, s, &[4, 3], 1000);
    let graph = g.finish();
    let run = Run::of(&graph, &[a, b, x, p], 0x0526_1700);
    run.accepted(&graph);
    let claim = &run.schedule.stages[0].claims[0];
    assert!(claim.split);
    assert!(
        claim.public.iter().all(|t| t.coefficient != [Vec::new()]),
        "no unit coefficient"
    );
}

/// RFC-0037 'Splits': a ready claim's pool value is its value less its public terms, each with its
/// coefficient. `requant(3·(x − Expand(p)))` and `requant(3·(x + c))`, `x` committed, are ready on
/// `x`, their terms of coefficients `−3` and `3`; both prove and verify, the pool `x`'s extension.
#[test]
fn a_ready_claim_with_a_scaled_public_term_proves_and_verifies() {
    for shift in [false, true] {
        let mut g = GraphBuilder::new();
        let x = small(&mut g, &[4, 3], ADVICE);
        let p = small(&mut g, &[3], PUBLIC);
        let three = g.constant(vec![3]).expect("a constant");
        let c = g.constant(vec![7, -2, 5]).expect("a constant");
        let virt = decl(&[4, 3], NodeKind::Virtual);
        let d = if shift {
            one(&mut g, Op::ConstAdd { c }, &[x], virt.clone())
        } else {
            let wide = one(&mut g, Op::Expand { axis: 0, len: 4 }, &[p], virt.clone());
            one(&mut g, Op::Sub, &[x, wide], virt.clone())
        };
        let e = one(&mut g, Op::ConstMul { c: three }, &[d], virt);
        requant(&mut g, e, &[4, 3], 1000);
        let graph = g.finish();
        let run = Run::of(&graph, &[x, p], 0x0526_1800);
        run.accepted(&graph);
        let claim = &run.schedule.stages[0].claims[0];
        assert!(!claim.split, "ready");
        assert_ne!(
            claim.public[0].coefficient,
            [Vec::new()],
            "a coefficient other than 1"
        );
    }
}

// ----- Refusals and the public mask ----------------------------------------------------------------

/// RFC-0072 'No reduction stage', RFC-0037 'The layer point': stage 1 draws the layer point at
/// which the leaf stage's per-layer claims stack. A graph with layers whose leaf claims all reach
/// committed tensors, `c_0 = requant(x)` and two layers of `requant(c_ℓ)`, has no stage 1 to draw
/// it, and is refused when its schedule is derived (`ERR-INTERNAL`), rather than failing in the
/// leaf stage's splits.
#[test]
fn a_graph_with_layers_and_no_stage_1_is_refused() {
    let mut b = GraphBuilder::new();
    let mut input = decl(&[4, 4], ADVICE);
    input.range = IntRange { lo: -100, hi: 100 };
    let x = b.input(input).expect("an input");
    let mut c = requant(&mut b, x, &[4, 4], 1000);
    let mut checkpoints = vec![c];
    for _ in 0..2 {
        c = requant(&mut b, c, &[4, 4], 1000);
        checkpoints.push(c);
    }
    let lowering = lowering_of(b.finish(), x, checkpoints);
    match derive_schedule(&lowering) {
        Err(error @ CompileError::Schedule { rule, .. }) => {
            assert_eq!(
                rule,
                "a graph with layers has a stage 1, which draws its layer point"
            );
            assert_eq!(error.code(), tessera_codec::error::ErrorCode::Internal);
        }
        other => panic!("refused, not {other:?}"),
    }
}

/// 05 §3 OBS-6: `public_value` reads a tensor only when the graph's [`PublicTensors`] marks it
/// public, so a caller cannot copy a committed or a virtual tensor's witness values into a public
/// table; and the mask must be the graph's. A public tensor's value is its padded extension.
#[test]
fn public_value_refuses_a_tensor_that_is_not_public() {
    let mut g = GraphBuilder::new();
    let x = small(&mut g, &[4, 3], ADVICE);
    let p = small(&mut g, &[3], PUBLIC);
    let wide = one(
        &mut g,
        Op::Expand { axis: 0, len: 4 },
        &[p],
        decl(&[4, 3], NodeKind::Virtual),
    );
    let y = one(
        &mut g,
        Op::Add,
        &[x, wide],
        decl(&[4, 3], NodeKind::Virtual),
    );
    let graph = g.finish();
    let witness = Engine::for_graph(&graph, &[])
        .expect("an engine")
        .evaluate(random_inputs(&graph, &[x, p], 0x0526_1900))
        .expect("a witness");
    let public = PublicTensors::of(&graph);
    let r = &random_points(4, 1, 0x0526_1901)[0];
    for secret in [x, y] {
        assert_eq!(
            public_value(&graph, &public, PublicSource::Tensor(secret), &witness, r),
            Err(StageError::Internal {
                rule: "a public table is of a public tensor"
            }),
            "tensor {}",
            secret.0
        );
    }
    let value = public_value(&graph, &public, PublicSource::Tensor(wide), &witness, r);
    let table = padded(&witness, &graph, wide);
    assert_eq!(value, Ok(evaluate_base::<F>(&table, r).expect("a point")));
    // The mask of another graph.
    let other = GraphBuilder::new().finish();
    assert_eq!(
        public_value(
            &graph,
            &PublicTensors::of(&other),
            PublicSource::Tensor(wide),
            &witness,
            r
        ),
        Err(StageError::Internal {
            rule: "a public mask is its graph's"
        })
    );
}

#[test]
fn stage_state_memory_covers_constructor_and_persisted_claims_parts_and_coordinates() {
    for name in MODELS {
        let (lowering, schedule, witness) = setup(name);
        let mut memory = None;
        let planning = allocation_counter::measure(|| {
            memory = Some(StageStateMemory::new(&lowering.graph, &schedule).unwrap());
        });
        assert_eq!(planning.count_total, 0);
        let memory = memory.unwrap();
        let mut held = None;
        let constructor = allocation_counter::measure(|| {
            held = Some(StageProver::new(&lowering, &schedule, &witness));
        });
        assert!(
            constructor.bytes_max <= memory.construction_peak_bytes(),
            "{name}: {constructor:?}, {memory:?}"
        );
        drop(held.take());
        let points = leaf_points(&schedule, 19);
        let retained = allocation_counter::measure(|| {
            let mut prover = StageProver::new(&lowering, &schedule, &witness);
            let mut t = transcript();
            let leaf = prover.leaf(&points).unwrap();
            absorb_leaf(&mut t, &leaf);
            drop(leaf);
            drop(prover.leaf_splits(&mut t).unwrap());
            for k in 1..schedule.stages.len() {
                drop(
                    prover
                        .prove_stage(&mut t, u16::try_from(k).unwrap())
                        .unwrap(),
                );
            }
            held = Some(prover);
        });
        assert!(
            u64::try_from(retained.bytes_current).unwrap() <= memory.retained_bytes(),
            "{name}: {retained:?}, {memory:?}"
        );
        let mut empty = schedule.clone();
        empty.stages.clear();
        let invalid = allocation_counter::measure(|| {
            assert_eq!(
                StageStateMemory::new(&lowering.graph, &empty),
                Err(StageError::Internal {
                    rule: "stage state has a compiled leaf stage"
                })
            );
        });
        assert_eq!(invalid.count_total, 0);
    }
}

#[test]
fn stage_source_geometry_never_reads_witness_values() {
    struct MetadataOnly;
    impl WitnessSource for MetadataOnly {
        fn values(&self, _: TensorId) -> Option<&[i64]> {
            panic!("public source geometry read witness values")
        }
    }
    for name in MODELS {
        let (lowering, schedule, _) = setup(name);
        let prover = StageProver::new(&lowering, &schedule, &MetadataOnly);
        for k in 1..schedule.stages.len() {
            let memory = prover
                .relation_source_memory(u16::try_from(k).unwrap(), |_| {
                    panic!("graph-only fixture has no native instances")
                })
                .unwrap();
            assert!(memory.metadata_bytes() > 0);
            assert!(memory.resident_input_bytes() > 0);
        }
        assert_eq!(
            prover.relation_source_memory(u16::MAX, |_| Ok(0)),
            Err(StageError::Internal {
                rule: "source memory names a compiled stage"
            })
        );
    }
}
