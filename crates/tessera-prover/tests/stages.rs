// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reduction stages of the conformance models (RFC-0010 'Stages', RFC-0037), proven over the
//! CPU engine's witness and checked by a verifier written here from RFC-0010 'Sumcheck messages'
//! and 'Batching within a stage' and RFC-0037 'Splits', 'The layer point' and 'Rotate'.
//!
//! The leaf stage is the gadgets' (#118). Here its claims are true values at random points, which
//! both sides bind under a test label, as the trees' messages will. The verifier ends with the
//! pool's values, and each must equal its committed tensors' extensions, computed from the witness.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use tessera_codec::{Ascii, DecodeLimits, Secret, SecretBuf, envelope, schema};
use tessera_compiler::schedule::{
    Claim, Coord, Factor, Layers, NodeRef, Part, Relation, Source, StageSchedule,
};
use tessera_compiler::{Lowering, PublicModel, derive_schedule, lower};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{
    CommitmentClass, GraphBuilder, IntRange, IntType, IntegerModel, NodeKind, Op, TensorDecl,
    TensorId,
};
use tessera_poly::{eq, eq_table, evaluate_base, less_than};
use tessera_prover::{StageProof, StageProver};
use tessera_sumcheck::SumcheckTranscript;
use tessera_tables::RopeTable;
use tessera_witness::{Engine, Witness};

type F = GoldilocksExt2;

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

// ----- A verifier from RFC-0010 and RFC-0037 ----------------------------------------------------

/// `Reject(Sumcheck { stage, round })` (04 §2), the leaf stage's splits at stage 0, round 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Reject {
    stage: u16,
    round: u32,
}

/// The verifier's state: the challenges, each stage's claims' and parts' values, and the pool.
struct Verifier<'a> {
    lowering: &'a Lowering,
    schedule: &'a StageSchedule,
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
    let mut v = Verifier {
        lowering,
        schedule,
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
        let Op::ConstMul { c } = self.lowering.graph.nodes[id.0 as usize].op else {
            panic!("a ConstMul");
        };
        F::from(Goldilocks::from_i64(
            self.lowering.graph.constants[c.0 as usize][0],
        ))
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

    /// The splits of stage `index` (RFC-0037 'Splits'): each split claim's parts' values, which
    /// must add up to its value, layer by layer for a leaf claim in the layers.
    fn splits(&mut self, index: usize, values: &[F], t: &mut Transcript) -> Result<(), Reject> {
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
            for (l, layer) in layers.iter().enumerate() {
                let value = self.claims[index][c][l];
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
        t.absorb_message(&format!("stage/{index}/split"), values);
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
    fn stage(&mut self, k: u16, proof: &StageProof, t: &mut Transcript) -> Result<(), Reject> {
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
            batched += power * self.relation(instance, e);
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
            Relation::MatMulWeight => 2 + usize::from(stacked),
            Relation::MatMulAct { .. } => {
                let heads = self.lowering.graph.tensors[node.inputs[0].0 as usize].shape[0];
                2 + usize::from(stacked || heads > 1)
            }
            Relation::Mul(_)
            | Relation::SumSquares
            | Relation::ScaleChannels
            | Relation::Booleanity => 3,
            Relation::Rotate => 2,
            Relation::Equality => match node.op {
                Op::Mul { .. } => 3,
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
        &self.lowering.graph.nodes[id.0 as usize]
    }

    /// An instance's relation at its point from its evaluations, public factors included.
    fn relation(&self, instance: &tessera_compiler::schedule::Instance, e: &[F]) -> F {
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
        let shape = |i: usize| &self.lowering.graph.tensors[node.inputs[i].0 as usize].shape;
        match instance.relation {
            Relation::MatMulWeight => {
                let layers =
                    eq(&target[target.len() - lambda..], &r[r.len() - lambda..]).expect("eq");
                layers * e[0] * e[1]
            }
            Relation::MatMulAct { .. } => {
                let out = &self.lowering.graph.tensors[node.outputs[0].0 as usize].shape;
                let (c, a) = (axis_vars(out[2]) as usize, axis_vars(out[1]) as usize);
                let inner = axis_vars(shape(0)[2]) as usize;
                eq(&target[c + a..], &r[inner..]).expect("eq") * e[0] * e[1]
            }
            Relation::Mul(_) => eq(&target, &r).expect("eq") * e[0] * e[1],
            Relation::SumSquares => {
                let last = axis_vars(*shape(0).last().expect("an axis")) as usize;
                eq(&target, &r[last..]).expect("eq") * e[0] * e[0]
            }
            Relation::Rotate => self.kernel(node, &target, &r, stacked) * e[0],
            Relation::ScaleChannels => panic!("no per-channel ConstMul in the conformance models"),
            Relation::Booleanity => eq(&target, &r).expect("eq") * e[0] * (F::ONE - e[0]),
            Relation::Equality => {
                let f = match node.op {
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
                        c * e[1]
                    }
                    Op::Add => e[1] + e[2],
                    Op::Sub => e[1] - e[2],
                    Op::Mul { .. } => e[1] * e[2],
                    _ => panic!("an equality zero-check's operation"),
                };
                eq(&target, &r).expect("eq") * (e[0] - f)
            }
        }
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
            .lowering
            .tables
            .iter()
            .find(|s| s.id == rope)
            .expect("the table");
        let table = RopeTable::from_spec(spec).expect("RoPE's constants");
        let x = &self.lowering.graph.tensors[node.inputs[0].0 as usize].shape;
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
fn padded(witness: &Witness, lowering: &Lowering, tensor: TensorId) -> Vec<Goldilocks> {
    let shape = &lowering.graph.tensors[tensor.0 as usize].shape;
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
    let v = Verifier {
        lowering,
        schedule,
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
                        .tensor(&lowering.graph, atom.kind, l)
                        .expect("a tensor");
                    let table = padded(witness, lowering, tensor);
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
fn lowering_of(graph: tessera_ir::Graph, x: TensorId, checkpoints: Vec<TensorId>) -> Lowering {
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
        let mean = one(&mut b, Op::Sum { axis: 0 }, &[c], virt(&[3, 4]));
        let spread = one(
            &mut b,
            Op::Expand { axis: 0, len: 5 },
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
