// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stage 1's bit zero-checks (RFC-0072 'The bit zero-checks', 'Testing Strategy'; RFC-0066
//! '`BitDecompose`', '`AssertEqual`', Lemma 1), proven by the stage prover and checked by the
//! verifier of this test, which evaluates every public factor itself by RFC-0072's closed forms:
//! `eq`, `1 − ψ̃`, `1 − LT(x_bit, n)`, the zero-extensions' products `Π_{t < k − k_o}(1 − r_t)` and
//! the weights `W̃_b`, `W̃_h`.
//!
//! - **The circuits' shapes.** A `BitDecompose { 8, 0 }` of committed bytes, a
//!   `BitDecompose { 32, 3 }` whose carry axis pads to 4, one with `k_h > k_b`, one of a virtual
//!   input and one of a public input, and an `AssertEqual` with one side public and with neither,
//!   prove and verify, each schedule's stage count the report's.
//! - **Adversarial.** Each cheating witness is rejected at stage 1's final check,
//!   `Reject(Sumcheck { stage: 1, round: n_1 })`, and accepted once exactly its check is removed
//!   from the schedule both parties follow, with the openings' pool its own committed tables'.
//!   The cheats commit padded positions through [`WitnessSource::padded`].
//! - **Alternative witnesses.** The bits are the unique passing witness of a `BitDecompose`, and
//!   equal tensors the unique passing pair of an `AssertEqual` ([gadget finals] 'Rejection order
//!   and alternatives').
//! - **The identity.** Recomposition summed over `{0, 1}^k` is RFC-0066's relation, and `W̃`'s
//!   closed form is the weights' extension.
//!
//! [gadget finals]: ../../../../docs/testing/gadget-finals.md

use super::*;
use tessera_compiler::schedule::{Instance as Planned, Kind};

// ----- The verifier's closed forms (RFC-0072 'The bit zero-checks') ----------------------------

/// `2^j` in the field.
fn two_to(j: u64) -> F {
    F::from(Goldilocks::from_u64_reduced(2)).pow(j)
}

/// `ψ̃(r) = Π_a LT(r_a, n_a)` of a tensor of `shape` (RFC-0072 'Real positions'), the last
/// axis's coordinates the lowest.
fn selector_at(shape: &[u32], r: &[F]) -> F {
    let mut at = 0;
    let mut value = F::ONE;
    for &len in shape.iter().rev() {
        let vars = axis_vars(len) as usize;
        value *= less_than(&r[at..at + vars], u64::from(len)).expect("LT");
        at += vars;
    }
    assert_eq!(at, r.len(), "a point of the tensor's variables");
    value
}

/// The zero-extension's product `Π_{t < pad}(1 − r_t)`.
fn zero_extension(r: &[F], pad: usize) -> F {
    r[..pad].iter().fold(F::ONE, |p, &x| p * (F::ONE - x))
}

/// `W̃(r)` on `k = r.len()` bit coordinates of an output of `bits` bits and `k_o = ⌈log2 bits⌉`
/// bit variables: `Π_{t < k − k_o}(1 − r_t)·Σ_{j < bits} 2^j·eq(s, j)`, `s` the high `k_o`
/// coordinates, low first; the sum is `Π_{t < k_o}(1 − s_t + s_t·2^{2^t})` when `bits = 2^{k_o}`.
fn weights_at(r: &[F], bits: u8) -> F {
    let k_o = axis_vars(u32::from(bits)) as usize;
    let pad = r.len() - k_o;
    let s = &r[pad..];
    let sum = if u32::from(bits) == 1 << k_o {
        (0..k_o).fold(F::ONE, |p, t| p * (F::ONE - s[t] + s[t] * two_to(1 << t)))
    } else {
        (0..u64::from(bits)).fold(F::ZERO, |sum, j| sum + two_to(j) * eq_at(s, j))
    };
    zero_extension(r, pad) * sum
}

impl Verifier<'_> {
    /// A bit zero-check at its point `target`, the instance's coordinates of the zero-check
    /// point, from its final point `r` and its evaluations `e` (RFC-0072's table): every public
    /// factor the verifier's own, a public input's value too.
    pub(super) fn bit_relation(
        &self,
        instance: &Planned,
        node: &tessera_ir::Node,
        target: &[F],
        r: &[F],
        e: &[F],
    ) -> F {
        let tensor = self
            .schedule
            .layers
            .tensor(self.graph, instance.kind, 0)
            .expect("a tensor");
        let shape = |t: TensorId| self.graph.tensors[t.0 as usize].shape.as_slice();
        let bit_vars = |bits: u8| axis_vars(u32::from(bits)) as usize;
        match instance.relation {
            Relation::InputPadding => {
                eq(target, r).expect("eq") * (F::ONE - selector_at(shape(tensor), r)) * e[0]
            }
            Relation::BitPadding => {
                let Op::BitDecompose { low, high } = node.op else {
                    panic!("a BitDecompose");
                };
                let bits = if node.outputs.get(1) == Some(&tensor) {
                    high
                } else {
                    low
                };
                let slot = less_than(&r[..bit_vars(bits)], u64::from(bits)).expect("LT");
                eq(target, r).expect("eq") * (F::ONE - slot) * e[0]
            }
            Relation::Recomposition => {
                let Op::BitDecompose { low, high } = node.op else {
                    panic!("a BitDecompose");
                };
                let k = bit_vars(low).max(bit_vars(high));
                let (r_i, r_y) = r.split_at(k);
                let mut next = 0;
                let x = self.read(node, 0, r_y, e, &mut next);
                let mut residual = two_to(k as u64).inverse().expect("2 ≠ 0") * x;
                residual -= weights_at(r_i, low) * zero_extension(r_i, k - bit_vars(low)) * e[next];
                if high > 0 {
                    residual -= two_to(u64::from(low))
                        * weights_at(r_i, high)
                        * zero_extension(r_i, k - bit_vars(high))
                        * e[next + 1];
                }
                // `eq` over `y`'s coordinates alone, the highest of the instance's.
                eq(&target[k..], r_y).expect("eq") * residual
            }
            Relation::AssertEqual => {
                let mut next = 0;
                let a = self.read(node, 0, r, e, &mut next);
                let b = self.read(node, 1, r, e, &mut next);
                eq(target, r).expect("eq") * selector_at(shape(node.inputs[0]), r) * (a - b)
            }
            _ => unreachable!("a bit zero-check"),
        }
    }
}

// ----- Graphs, witnesses and runs ---------------------------------------------------------------

/// A graph input of `shape` and `kind` declared in `[lo, hi]`.
fn declared(g: &mut GraphBuilder, shape: &[u32], kind: NodeKind, lo: i128, hi: i128) -> TensorId {
    let input = TensorDecl {
        shape: shape.to_vec(),
        ty: IntType {
            bits: 64,
            signed: true,
        },
        range: IntRange { lo, hi },
        kind,
    };
    g.input(input).expect("an input")
}

/// A bit tensor of `shape`, committed advice.
fn bit_tensor(shape: &[u32]) -> TensorDecl {
    let mut bits = decl(shape, ADVICE);
    bits.range = IntRange { lo: 0, hi: 1 };
    bits
}

/// `BitDecompose { low, high }` of `x`, of `shape`: `b`, and `h` when `high > 0`.
fn decompose(
    g: &mut GraphBuilder,
    x: TensorId,
    shape: &[u32],
    low: u8,
    high: u8,
) -> (TensorId, Option<TensorId>) {
    let with = |bits: u8| {
        let mut s = shape.to_vec();
        s.push(u32::from(bits));
        bit_tensor(&s)
    };
    let mut outputs = vec![with(low)];
    if high > 0 {
        outputs.push(with(high));
    }
    let out = g
        .node(Op::BitDecompose { low, high }, &[x], outputs)
        .expect("a decomposition");
    (out[0], out.get(1).copied())
}

/// The index of `at`, a position of a tensor of `shape`, in its padded table: the last axis the
/// lowest bits, each axis its own block of `⌈log2 n_a⌉`.
pub(super) fn index(shape: &[u32], at: &[u32]) -> usize {
    let mut index = 0;
    let mut shift = 0;
    for (&len, &i) in shape.iter().zip(at).rev() {
        index |= (i as usize) << shift;
        shift += axis_vars(len);
    }
    index
}

/// `tensor`'s honest table over its padded shape, its padded positions 0.
pub(super) fn table_of(witness: &Witness, graph: &Graph, tensor: TensorId) -> Vec<i64> {
    padded(witness, graph, tensor)
        .into_iter()
        .map(|v| v.to_centered_i128() as i64)
        .collect()
}

/// A prover that commits the tables `tables` holds over their padded shapes, their padded
/// positions among them, and the honest witness's other tensors.
pub(super) struct Cheat<'a> {
    pub(super) witness: &'a Witness,
    pub(super) tables: Vec<(TensorId, Vec<i64>)>,
}

impl WitnessSource for Cheat<'_> {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        self.witness.values(tensor)
    }

    fn padded(&self, tensor: TensorId) -> Option<&[i64]> {
        self.tables
            .iter()
            .find(|(t, _)| *t == tensor)
            .map(|(_, values)| values.as_slice())
    }
}

/// The verifier's public data: the public graph inputs' values, and every other public tensor,
/// which by the closure is a function of them alone (RFC-0072 'Public tensors'), from the honest
/// run on the same public inputs. A test that changes a public input changes one no other public
/// tensor reads.
struct PublicData<'a> {
    inputs: &'a [(TensorId, Vec<i64>)],
    witness: &'a Witness,
    public: Vec<bool>,
}

impl WitnessSource for PublicData<'_> {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        if let Some((_, values)) = self.inputs.iter().find(|(t, _)| *t == tensor) {
            return Some(values);
        }
        if !self.public.get(tensor.0 as usize).copied().unwrap_or(false) {
            return None;
        }
        self.witness.values(tensor)
    }
}

/// A graph's honest run on given inputs: its schedule, the witness, the public inputs' values
/// and the leaf points.
struct Bits {
    graph: Graph,
    schedule: StageSchedule,
    witness: Witness,
    public: Vec<(TensorId, Vec<i64>)>,
    points: Vec<Vec<F>>,
}

impl Bits {
    /// The run of `graph` on `inputs`, every graph input's values: its schedule's stage count is
    /// the report's, and the honest proof verifies, its pool the committed tensors' extensions.
    fn of(graph: Graph, inputs: Vec<(TensorId, Vec<i64>)>) -> Self {
        let schedule = derive_graph_schedule(&graph, &[], &[], None).expect("a schedule");
        assert_eq!(
            schedule.stage_count(),
            tessera_compiler::stage_count(&graph) as usize,
            "the report's stage count"
        );
        let flags = public_tensors(&graph);
        let public = inputs
            .iter()
            .filter(|(t, _)| flags[t.0 as usize])
            .cloned()
            .collect();
        let witness = Engine::for_graph(&graph, &[])
            .expect("an engine")
            .evaluate(
                inputs
                    .into_iter()
                    .map(|(t, v)| (t, v.into_iter().collect()))
                    .collect(),
            )
            .expect("an honest witness");
        let points = leaf_points(&schedule, 0x0526_2000);
        let run = Self {
            graph,
            schedule,
            witness,
            public,
            points,
        };
        let honest = run.prove(&run.schedule, &run.witness);
        let pool = run
            .verify(&run.schedule, &honest, &run.public)
            .expect("the honest proof verifies");
        let coords = coords_after(&run.schedule, &honest, &run.points);
        assert_eq!(
            pool,
            pool_of_graph(&coords, &run.graph, &run.schedule, &run.witness),
            "the pool is the committed tensors' extensions"
        );
        // Stage 1 has degree 3: its booleanity checks' (RFC-0072 'Degree').
        for round in &honest.stages[0].sumcheck.rounds {
            assert_eq!(round.len(), 3, "stage 1's degree");
        }
        run
    }

    fn prove(&self, schedule: &StageSchedule, source: &dyn WitnessSource) -> Proof {
        prove_graph(&self.graph, &[], schedule, source, &self.points).0
    }

    fn verify(
        &self,
        schedule: &StageSchedule,
        proof: &Proof,
        public: &[(TensorId, Vec<i64>)],
    ) -> Result<Vec<F>, Reject> {
        let values = PublicData {
            inputs: public,
            witness: &self.witness,
            public: public_tensors(&self.graph),
        };
        let publics = Publics {
            values: &values,
            shift: Shift::Extension,
            kernel: Kernel::Closed,
        };
        verify_public(&self.graph, &[], schedule, proof, &self.points, publics)
    }

    /// Stage 1's final check, `Reject(Sumcheck { stage: 1, round: n_1 })`.
    fn stage_1(&self) -> Reject {
        Reject {
            stage: 1,
            round: self.schedule.stages[1].rounds,
        }
    }

    /// The honest table of `tensor`, with `edits` at the positions they name.
    fn edited(&self, tensor: TensorId, edits: &[(&[u32], i64)]) -> (TensorId, Vec<i64>) {
        let shape = &self.graph.tensors[tensor.0 as usize].shape;
        let mut table = table_of(&self.witness, &self.graph, tensor);
        for &(at, value) in edits {
            table[index(shape, at)] = value;
        }
        (tensor, table)
    }

    /// The index of stage 1's instance of `relation` on `tensor`'s kind.
    fn check(&self, relation: Relation, tensor: TensorId) -> usize {
        let kind = self.schedule.layers.outside(tensor).0;
        self.schedule.stages[1]
            .instances
            .iter()
            .position(|i| i.relation == relation && i.kind == kind)
            .unwrap_or_else(|| panic!("a {relation:?} on tensor {}", tensor.0))
    }

    /// Each cheat, committing its tables: every one rejected at stage 1's final check under the
    /// schedule, then each accepted under it with exactly stage 1's instance of its check, a
    /// relation on a tensor, taken out, the openings' pool then the cheat's own tables'
    /// extensions. The rejections come first, so that a schedule without a check fails at the
    /// cheat it lets through.
    fn caught(&self, cheats: &[Cheating<'_>]) {
        for (name, tables, _) in cheats {
            let cheat = Cheat {
                witness: &self.witness,
                tables: tables.clone(),
            };
            let bad = self.prove(&self.schedule, &cheat);
            assert_eq!(
                self.verify(&self.schedule, &bad, &self.public),
                Err(self.stage_1()),
                "{name}: rejected at stage 1's final check"
            );
        }
        for (name, tables, check) in cheats {
            self.accepted_without(name, tables.clone(), *check);
        }
    }

    /// The cheat committing `tables`, accepted with exactly stage 1's instance of `check` taken
    /// out, the openings' pool then the cheat's own tables' extensions.
    fn accepted_without(
        &self,
        name: &str,
        tables: Vec<(TensorId, Vec<i64>)>,
        check: (Relation, TensorId),
    ) {
        let cheat = Cheat {
            witness: &self.witness,
            tables,
        };
        let without = without(&self.schedule, self.check(check.0, check.1));
        // Slice 2's cheats each break one check whose claims all reach the pool: exactly one
        // instance goes.
        let count = |s: &StageSchedule| s.stages.iter().map(|s| s.instances.len()).sum::<usize>();
        assert_eq!(
            count(&without) + 1,
            count(&self.schedule),
            "{name}: exactly one instance removed"
        );
        let passed = self.prove(&without, &cheat);
        let pool = self
            .verify(&without, &passed, &self.public)
            .unwrap_or_else(|reject| panic!("{name}: accepted without its check, not {reject:?}"));
        let coords = coords_after(&without, &passed, &self.points);
        let opened = pool_of_tables(&coords, &self.graph, &without, &|t| match cheat.padded(t) {
            Some(values) => values.iter().map(|&v| Goldilocks::from_i64(v)).collect(),
            None => padded(&self.witness, &self.graph, t),
        });
        assert_eq!(
            pool, opened,
            "{name}: the openings accept the cheat's tables"
        );
    }
}

/// A cheat: its name, the tables it commits, and the check, a relation on a tensor, that rejects
/// it.
type Cheating<'a> = (&'a str, Vec<(TensorId, Vec<i64>)>, (Relation, TensorId));

/// `schedule` with stage 1's instance `removed` taken out: the protocol without that one check.
/// Its claims go with it, and with them their pool claims and the instances of later stages
/// that prove their virtual parts, with those instances' own claims in turn: no other claim
/// reads them, so nothing else changes. Every index after a removed one moves down. Every stage
/// keeps its rounds and stage 1 its zero-check point: the removed instances are no stage's
/// only largest. Removing a check whose claims all reach the pool, as every cheat on a committed
/// input's bits here does, takes out that one instance alone; a SHA-256 sum's recomposition also
/// takes out the chain that proves its claim on the sum's value ([`super::sha256`]).
pub(super) fn without(schedule: &StageSchedule, removed: usize) -> StageSchedule {
    // The instances, claims and pool claims removed, stage by stage.
    let mut instances: Vec<Vec<bool>> = schedule
        .stages
        .iter()
        .map(|s| vec![false; s.instances.len()])
        .collect();
    let mut claims: Vec<Vec<bool>> = schedule
        .stages
        .iter()
        .map(|s| vec![false; s.claims.len()])
        .collect();
    let mut pool = vec![false; schedule.pool.len()];
    instances[1][removed] = true;
    for (k, stage) in schedule.stages.iter().enumerate() {
        for (i, instance) in stage.instances.iter().enumerate() {
            if !instances[k][i] {
                continue;
            }
            for c in instance.claims.clone() {
                claims[k][c] = true;
                for part in &stage.claims[c].parts {
                    match *part {
                        Part::Instance { instance, .. } => instances[k + 1][instance] = true,
                        Part::Pool { claim, .. } => pool[claim] = true,
                    }
                }
            }
        }
    }
    // An index's new value: the kept indices before it.
    let kept = |removed: &[bool], index: usize| removed[..index].iter().filter(|r| !**r).count();
    let mut s = schedule.clone();
    for (k, stage) in s.stages.iter_mut().enumerate() {
        let old = core::mem::take(&mut stage.instances);
        for (i, mut instance) in old.into_iter().enumerate() {
            if instances[k][i] {
                continue;
            }
            let start = kept(&claims[k], instance.claims.start);
            instance.claims = start..start + instance.claims.len();
            if let Source::Part { claim, .. } = &mut instance.source {
                *claim = kept(&claims[k - 1], *claim);
            }
            stage.instances.push(instance);
        }
        let old = core::mem::take(&mut stage.claims);
        for (c, mut claim) in old.into_iter().enumerate() {
            if claims[k][c] {
                continue;
            }
            for part in &mut claim.parts {
                match part {
                    Part::Instance { instance, .. } => {
                        *instance = kept(&instances[k + 1], *instance)
                    }
                    Part::Pool { claim, .. } => *claim = kept(&pool, *claim),
                }
            }
            stage.claims.push(claim);
        }
    }
    s.pool = s
        .pool
        .into_iter()
        .enumerate()
        .filter(|(p, _)| !pool[*p])
        .map(|(_, mut claim)| {
            claim.claim = kept(&claims[usize::from(claim.stage)], claim.claim);
            claim
        })
        .collect();
    for stage in &s.stages[1..] {
        let most = |zero: bool| {
            stage
                .instances
                .iter()
                .filter(|i| !zero || i.source == Source::ZeroCheck)
                .map(|i| i.vars)
                .max()
                .unwrap_or(0)
        };
        assert_eq!(
            stage.rounds,
            most(false),
            "stage {}'s rounds stay",
            stage.index
        );
        if stage.index == 1 {
            assert_eq!(
                stage.zero_vars,
                most(true),
                "stage 1's zero-check point stays"
            );
        }
    }
    s
}

const U8: (i128, i128) = (0, 255);

/// A tensor of `[3]` and `kind` declared in `[0, 2^35)`: a `BitDecompose { 32, 3 }`'s input.
fn below_2_35(kind: NodeKind) -> TensorDecl {
    let mut x = decl(&[3], kind);
    x.ty = IntType {
        bits: 64,
        signed: true,
    };
    x.range = IntRange {
        lo: 0,
        hi: (1 << 35) - 1,
    };
    x
}

/// `u`, committed bytes of `[3, 2]`, decomposed by `BitDecompose { 8, 0 }`: its padded rows are
/// the input padding check's.
fn bytes(values: &[i64]) -> (Bits, TensorId, TensorId) {
    let mut g = GraphBuilder::new();
    let u = declared(&mut g, &[3, 2], ADVICE, U8.0, U8.1);
    let (b, _) = decompose(&mut g, u, &[3, 2], 8, 0);
    let run = Bits::of(g.finish(), vec![(u, values.to_vec())]);
    (run, u, b)
}

/// The bits of `value`: `low` of them, then `high`.
fn bits_of(value: i64, low: u8, high: u8) -> (Vec<i64>, Vec<i64>) {
    let bit = |j: u32| (value >> j) & 1;
    (
        (0..u32::from(low)).map(bit).collect(),
        (0..u32::from(high))
            .map(|j| bit(u32::from(low) + j))
            .collect(),
    )
}

// ----- The circuits' shapes ---------------------------------------------------------------------

/// The instances' relations and kinds of stage 1, in order.
fn relations(run: &Bits) -> Vec<(Relation, Kind)> {
    run.schedule.stages[1]
        .instances
        .iter()
        .map(|i| (i.relation, i.kind))
        .collect()
}

/// RFC-0072 'The bit zero-checks': a `BitDecompose { 8, 0 }` of committed bytes with padded rows
/// is its input's padding check, its bits' booleanity and its recomposition, in that order, all
/// of whose claims reach the pool; a `BitDecompose { 32, 3 }` adds the carry's booleanity and
/// its padding to 4 slots; with `k_h > k_b`, `b` is the zero-extended output. Each proves and
/// verifies (in [`Bits::of`]).
#[test]
fn bit_decompositions_prove_and_verify() {
    use Relation::{BitPadding, Booleanity, InputPadding, Recomposition};
    let (run, u, b) = bytes(&[17, 255, 0, 128, 2, 99]);
    let t = Kind::Tensor;
    assert_eq!(
        relations(&run),
        [
            (InputPadding, t(u)),
            (Booleanity, t(b)),
            (Recomposition, t(b))
        ]
    );
    assert_eq!(
        run.schedule.stage_count(),
        2,
        "every claim reaches the pool"
    );
    // Carries: x committed below 2^35.
    let mut g = GraphBuilder::new();
    let x = declared(&mut g, &[3], ADVICE, 0, (1 << 35) - 1);
    let (b, h) = decompose(&mut g, x, &[3], 32, 3);
    let h = h.expect("carries");
    let run = Bits::of(
        g.finish(),
        vec![(x, vec![(2 << 32) + 12_345, (7 << 32) + 0xFFFF_FFFF, 5])],
    );
    assert_eq!(
        relations(&run),
        [
            (InputPadding, t(x)),
            (Booleanity, t(b)),
            (Booleanity, t(h)),
            (BitPadding, t(h)),
            (Recomposition, t(b)),
        ]
    );
    // `BitDecompose { 2, 8 }`: `k_h = 3 > k_b = 1`, so `b` is zero-extended, and recomposition's
    // kind is `h`, whose bit axis has its `k` variables.
    let mut g = GraphBuilder::new();
    let u = declared(&mut g, &[3], ADVICE, 0, 1023);
    let (b, h) = decompose(&mut g, u, &[3], 2, 8);
    let h = h.expect("high bits");
    let run = Bits::of(g.finish(), vec![(u, vec![1023, 513, 6])]);
    assert_eq!(
        relations(&run),
        [
            (InputPadding, t(u)),
            (Booleanity, t(b)),
            (Booleanity, t(h)),
            (Recomposition, t(h)),
        ]
    );
    // Recomposition's claims: `u` at `r'_y`, `b` at the high `k_b` bit coordinates with `r'_y`,
    // `h` at all of them, its booleanity's point.
    let stage = &run.schedule.stages[1];
    let recomposition = &stage.instances[3];
    let r = &recomposition.challenges;
    let points: Vec<&[Coord]> = stage.claims[recomposition.claims.clone()]
        .iter()
        .map(|c| c.point.as_slice())
        .collect();
    assert_eq!(points, [&r[3..], &r[2..], &r[..]]);
    assert_eq!(stage.claims[stage.instances[2].claims.start].point, *r);
}

/// RFC-0072 'Its claims': recomposition's claim on a virtual `x`, the product of committed bytes
/// and a public scale, goes to stage 2, which proves the product; on a public `x`, the verifier
/// evaluates it and the proof sends nothing for it. Both prove and verify, and a wrong result bit
/// is rejected at stage 1.
#[test]
fn a_decomposition_of_a_virtual_or_public_input_proves_and_verifies() {
    for public in [false, true] {
        let mut g = GraphBuilder::new();
        let kind = if public { PUBLIC } else { ADVICE };
        let u = declared(&mut g, &[3], kind, 0, (1 << 17) - 1);
        if !public {
            decompose(&mut g, u, &[3], 17, 0);
        }
        let p = declared(&mut g, &[3], PUBLIC, 0, (1 << 17) - 1);
        let mul = product(tessera_ir::Broadcast::Trailing);
        let x = one(&mut g, mul, &[u, p], below_2_35(NodeKind::Virtual));
        let (b, _) = decompose(&mut g, x, &[3], 32, 3);
        let values = vec![(u, vec![131_071, 99_999, 3]), (p, vec![131_000, 77_777, 5])];
        let run = Bits::of(g.finish(), values);
        let stage = &run.schedule.stages[1];
        let recomposition = stage
            .instances
            .iter()
            .find(|i| i.relation == Relation::Recomposition && i.kind == Kind::Tensor(b))
            .expect("x's recomposition");
        let kinds: Vec<Kind> = stage.claims[recomposition.claims.clone()]
            .iter()
            .map(|c| c.kind)
            .collect();
        let h = run.graph.nodes.last().expect("x's decomposition").outputs[1];
        if public {
            assert_eq!(kinds, [Kind::Tensor(b), Kind::Tensor(h)], "no claim on x");
            assert_eq!(run.schedule.stage_count(), 2);
        } else {
            assert_eq!(kinds[0], Kind::Tensor(x));
            let claim = &stage.claims[recomposition.claims.start];
            assert!(
                matches!(claim.parts[..], [Part::Instance { .. }]),
                "x's claim is the product's instance of stage 2"
            );
            assert_eq!(run.schedule.stage_count(), 3);
        }
        // x = 15 at [2]: its bit 4 set.
        let bad = run.edited(b, &[(&[2, 4], 1)]);
        let cheat = Cheat {
            witness: &run.witness,
            tables: vec![bad],
        };
        assert_eq!(
            run.verify(
                &run.schedule,
                &run.prove(&run.schedule, &cheat),
                &run.public
            ),
            Err(run.stage_1()),
            "public {public}: a wrong result bit"
        );
    }
}

/// RFC-0066 '`AssertEqual`', RFC-0072 'The bit zero-checks': one side public sends the other's
/// evaluation alone; neither public sends `ã` then `b̃`, each to its destination, a virtual side
/// to the next stage; both public send nothing, the verifier rejecting a side one off. Each
/// proves and verifies.
#[test]
fn assert_equal_proves_and_verifies() {
    let values = vec![7, 0, 255, 3, 128, 64];
    // One side public: the binding of a digest.
    let mut g = GraphBuilder::new();
    let u = declared(&mut g, &[3, 2], ADVICE, U8.0, U8.1);
    decompose(&mut g, u, &[3, 2], 8, 0);
    let p = declared(&mut g, &[3, 2], PUBLIC, U8.0, U8.1);
    g.node(Op::AssertEqual, &[u, p], Vec::new())
        .expect("a binding");
    let run = Bits::of(g.finish(), vec![(u, values.clone()), (p, values.clone())]);
    let stage = &run.schedule.stages[1];
    let check = stage.instances.last().expect("the binding");
    assert_eq!((check.relation, check.vars), (Relation::AssertEqual, 3));
    assert_eq!(check.claims.len(), 1, "the public side sends nothing");
    assert_eq!(stage.claims[check.claims.start].kind, Kind::Tensor(u));
    // Neither public, one side virtual: `u·q` against committed `v`.
    let mut g = GraphBuilder::new();
    let u = declared(&mut g, &[3, 2], ADVICE, U8.0, U8.1);
    decompose(&mut g, u, &[3, 2], 8, 0);
    let v = declared(&mut g, &[3, 2], ADVICE, U8.0, U8.1);
    decompose(&mut g, v, &[3, 2], 8, 0);
    let q = declared(&mut g, &[3, 2], PUBLIC, 0, 1);
    let mul = product(tessera_ir::Broadcast::Trailing);
    let y = one(&mut g, mul, &[u, q], decl(&[3, 2], NodeKind::Virtual));
    g.node(Op::AssertEqual, &[y, v], Vec::new())
        .expect("an equality");
    let mask = vec![1, 0, 1, 1, 0, 1];
    let masked: Vec<i64> = values.iter().zip(&mask).map(|(a, b)| a * b).collect();
    let run = Bits::of(
        g.finish(),
        vec![(u, values.clone()), (v, masked), (q, mask)],
    );
    let stage = &run.schedule.stages[1];
    let check = stage.instances.last().expect("the equality");
    let claims = &stage.claims[check.claims.clone()];
    let kinds: Vec<Kind> = claims.iter().map(|c| c.kind).collect();
    assert_eq!(kinds, [Kind::Tensor(y), Kind::Tensor(v)]);
    assert!(matches!(claims[0].parts[..], [Part::Instance { .. }]));
    assert!(matches!(claims[1].parts[..], [Part::Pool { .. }]));
    assert_eq!(run.schedule.stage_count(), 3);
    // Both public: one instance, which sends nothing; the verifier evaluates both sides.
    let mut g = GraphBuilder::new();
    let p = declared(&mut g, &[3, 2], PUBLIC, U8.0, U8.1);
    let q = declared(&mut g, &[3, 2], PUBLIC, U8.0, U8.1);
    g.node(Op::AssertEqual, &[p, q], Vec::new())
        .expect("an equality");
    let run = Bits::of(g.finish(), vec![(p, values.clone()), (q, values)]);
    let check = &run.schedule.stages[1].instances[0];
    assert_eq!(check.claims.len(), 0, "no evaluation");
    let mut changed = run.public.clone();
    changed[1].1[2] -= 1;
    let honest = run.prove(&run.schedule, &run.witness);
    assert!(honest.stages[0].sumcheck.final_evals.is_empty());
    assert_eq!(
        run.verify(&run.schedule, &honest, &changed),
        Err(run.stage_1()),
        "a public side one off"
    );
}

/// RFC-0072 'Messages': the bit zero-checks add no label. A graph of them proves on a transcript
/// that records every label, and each is one of RFC-0014's rows 6 for stages 0 to 2.
#[test]
fn the_bit_zero_checks_add_no_label() {
    struct Recording {
        inner: Transcript,
        labels: Vec<String>,
    }
    impl SumcheckTranscript<F> for Recording {
        fn absorb_message(&mut self, label: &str, message: &[F]) {
            self.labels.push(label.to_owned());
            self.inner.absorb_message(label, message);
        }
        fn challenge(&mut self, label: &str) -> F {
            self.labels.push(label.to_owned());
            self.inner.challenge(label)
        }
    }
    let mut g = GraphBuilder::new();
    let u = declared(&mut g, &[3], ADVICE, 0, (1 << 17) - 1);
    decompose(&mut g, u, &[3], 17, 0);
    let p = declared(&mut g, &[3], PUBLIC, 0, (1 << 17) - 1);
    let mul = product(tessera_ir::Broadcast::Trailing);
    let x = one(&mut g, mul, &[u, p], below_2_35(NodeKind::Virtual));
    decompose(&mut g, x, &[3], 32, 3);
    g.node(Op::AssertEqual, &[u, p], Vec::new())
        .expect("a binding");
    let values = vec![99, 1, 131_071];
    let run = Bits::of(g.finish(), vec![(u, values.clone()), (p, values)]);
    let mut t = Recording {
        inner: transcript(),
        labels: Vec::new(),
    };
    let mut prover = StageProver::for_graph(&run.graph, &[], &run.schedule, &run.witness);
    prover.leaf(&run.points).expect("an empty leaf stage");
    prover.leaf_splits(&mut t).expect("no split");
    for k in 1..run.schedule.stages.len() {
        let k = u16::try_from(k).expect("small");
        prover.prove_stage(&mut t, k).expect("a stage");
    }
    let known = |label: &str| {
        let parts: Vec<&str> = label.split('/').collect();
        match parts[..] {
            ["stage", "0", "split"] => true,
            ["stage", k, ref rest @ ..] if k == "1" || k == "2" => {
                matches!(
                    rest,
                    ["gamma"] | ["evals"] | ["split"] | ["round", _] | ["r", _]
                ) || (k == "1" && matches!(rest, ["zero", _]))
            }
            _ => false,
        }
    };
    assert!(t.labels.iter().any(|l| l == "stage/1/evals"));
    for label in &t.labels {
        assert!(known(label), "{label}: RFC-0014's row 6");
    }
}

// ----- Adversarial: each check's violation, and the control without it ---------------------------

/// RFC-0072 'Testing Strategy' 'The bit zero-checks', on committed bytes: a wrong bit, a
/// non-Boolean bit with recomposition holding, a committed byte of 256 (with Boolean bits, which
/// cannot recompose it, and with bits that recompose it, one of them 2), and a committed input's
/// padded position set non-zero with bits that recompose it. Each is rejected at stage 1's final
/// check, `Reject(Sumcheck { stage: 1, round: n_1 })`, and accepted with exactly the check it
/// violates removed.
#[test]
fn a_cheat_on_committed_bytes_fails_exactly_its_check() {
    let values = [17, 255, 0, 128, 2, 99];
    let (run, u, b) = bytes(&values);
    let padding = (Relation::InputPadding, u);
    let booleanity = (Relation::Booleanity, b);
    let recomposition = (Relation::Recomposition, b);
    // 17 = 0b10001 at [0, 0]: its bit 3 set.
    let wrong = run.edited(b, &[(&[0, 0, 3], 1)]);
    // 2 at [2, 0]: bit 0 of value 2 and bit 1 of 0 recompose it.
    let non_boolean = run.edited(b, &[(&[2, 0, 0], 2), (&[2, 0, 1], 0)]);
    // 256 at [0, 1], whose eight Boolean bits are 0: they cannot recompose it.
    let byte = run.edited(u, &[(&[0, 1], 256)]);
    let zeros: Vec<(&[u32], i64)> = Vec::new();
    let mut bits = run.edited(b, &zeros);
    let shape = &run.graph.tensors[b.0 as usize].shape;
    for j in 0..8 {
        bits.1[index(shape, &[0, 1, j])] = 0;
    }
    // ... or bits that recompose it, `2·2^7`: booleanity fails.
    let mut recomposing = bits.clone();
    recomposing.1[index(shape, &[0, 1, 7])] = 2;
    // Padded row 3 of `u` holds 5, and its bits recompose 5: the input padding check alone.
    let padded_input = run.edited(u, &[(&[3, 0], 5)]);
    let padded_bits = run.edited(b, &[(&[3, 0, 0], 1), (&[3, 0, 2], 1)]);
    run.caught(&[
        ("a wrong bit", vec![wrong], recomposition),
        (
            "a non-Boolean bit that recomposes",
            vec![non_boolean],
            booleanity,
        ),
        (
            "a byte of 256, Boolean bits",
            vec![byte.clone(), bits],
            recomposition,
        ),
        (
            "a byte of 256, bits that recompose it",
            vec![byte, recomposing],
            booleanity,
        ),
        (
            "a committed input's padded position",
            vec![padded_input, padded_bits],
            padding,
        ),
    ]);
}

/// RFC-0072 'Testing Strategy' 'The bit zero-checks', on a `BitDecompose { 32, 3 }` of committed
/// sums below `2^35`, its carry's axis padded to 4: a wrong carry bit, a carry made up by a
/// non-Boolean bit, a padded carry slot set to 1, and a wrong result bit, each rejected at stage
/// 1's final check and accepted with exactly its check removed.
#[test]
fn a_cheat_on_carries_fails_exactly_its_check() {
    let mut g = GraphBuilder::new();
    let x = declared(&mut g, &[3], ADVICE, 0, (1 << 35) - 1);
    let (b, h) = decompose(&mut g, x, &[3], 32, 3);
    let h = h.expect("carries");
    // Carries 0b010, 0b111 and 0.
    let values = vec![(2 << 32) + 12_345, (7 << 32) + 0xFFFF_FFFF, 5];
    let run = Bits::of(g.finish(), vec![(x, values)]);
    let recomposition = (Relation::Recomposition, b);
    let carry = run.edited(h, &[(&[0, 2], 1)]);
    let made_up = run.edited(h, &[(&[0, 0], 2), (&[0, 1], 0)]);
    let slot = run.edited(h, &[(&[1, 3], 1)]);
    // 5 = 0b101 at [2]: its bit 1 set.
    let result = run.edited(b, &[(&[2, 1], 1)]);
    run.caught(&[
        ("a wrong carry bit", vec![carry], recomposition),
        (
            "a carry made up by a non-Boolean bit",
            vec![made_up],
            (Relation::Booleanity, h),
        ),
        (
            "a padded carry slot set to 1",
            vec![slot],
            (Relation::BitPadding, h),
        ),
        ("a wrong result bit", vec![result], recomposition),
    ]);
}

/// A wrong bit of a `BitDecompose { 2, 8 }`, whose low bits are zero-extended on their high
/// coordinate: rejected at stage 1, accepted without recomposition.
#[test]
fn a_cheat_on_zero_extended_bits_fails_recomposition() {
    let mut g = GraphBuilder::new();
    let u = declared(&mut g, &[3], ADVICE, 0, 1023);
    let (b, h) = decompose(&mut g, u, &[3], 2, 8);
    let h = h.expect("high bits");
    let run = Bits::of(g.finish(), vec![(u, vec![1023, 513, 6])]);
    let recomposition = (Relation::Recomposition, h);
    // 6 = 0b110 at [2]: its bit 0 set.
    let low = run.edited(b, &[(&[2, 0], 1)]);
    let high = run.edited(h, &[(&[1, 7], 0)]);
    run.caught(&[
        ("a wrong low bit", vec![low], recomposition),
        ("a wrong high bit", vec![high], recomposition),
    ]);
}

/// RFC-0072 'Testing Strategy': an `AssertEqual` side one off, its committed side's bits
/// recomputed so that its decomposition holds, against a public side and against a committed
/// one: rejected at stage 1's final check, and accepted without the equality's check. A public
/// side one off on the verifier's side is rejected too.
#[test]
fn an_assert_equal_side_one_off_fails_its_check() {
    let values = vec![7, 0, 255, 3, 128, 64];
    for public in [true, false] {
        let mut g = GraphBuilder::new();
        let u = declared(&mut g, &[3, 2], ADVICE, U8.0, U8.1);
        let (bu, _) = decompose(&mut g, u, &[3, 2], 8, 0);
        let kind = if public { PUBLIC } else { ADVICE };
        let v = declared(&mut g, &[3, 2], kind, U8.0, U8.1);
        if !public {
            decompose(&mut g, v, &[3, 2], 8, 0);
        }
        g.node(Op::AssertEqual, &[u, v], Vec::new())
            .expect("an equality");
        let run = Bits::of(g.finish(), vec![(u, values.clone()), (v, values.clone())]);
        // The equality's instance names its first side, `u`.
        let equality = (Relation::AssertEqual, u);
        // 3 at [1, 1] becomes 4: bits 0 and 1 cleared, bit 2 set.
        let off = run.edited(u, &[(&[1, 1], 4)]);
        let bits = run.edited(bu, &[(&[1, 1, 0], 0), (&[1, 1, 1], 0), (&[1, 1, 2], 1)]);
        let name = format!("a committed side one off, the other public {public}");
        run.caught(&[(name.as_str(), vec![off, bits], equality)]);
        if public {
            let mut changed = run.public.clone();
            changed[0].1[5] += 1;
            let honest = run.prove(&run.schedule, &run.witness);
            assert_eq!(
                run.verify(&run.schedule, &honest, &changed),
                Err(run.stage_1()),
                "the public side one off"
            );
        }
    }
}

// ----- Alternative witnesses (gadget finals 'Rejection order and alternatives') -----------------

/// RFC-0066 Lemma 1: of every Boolean assignment of a `BitDecompose { 2, 3 }`'s five bits and its
/// carry's padded slot at one position, only the honest one passes, and so do none of the
/// non-Boolean assignments that recompose it; the rest of the witness is honest. Each other is
/// rejected at stage 1's final check, none at a gadget (`Reject(Gadget)`).
#[test]
fn the_bits_are_the_unique_passing_witness() {
    let mut g = GraphBuilder::new();
    let u = declared(&mut g, &[2], ADVICE, 0, 31);
    let (b, h) = decompose(&mut g, u, &[2], 2, 3);
    let h = h.expect("high bits");
    let run = Bits::of(g.finish(), vec![(u, vec![13, 22])]);
    let (honest_b, honest_h) = bits_of(13, 2, 3);
    let mut accepted = 0;
    // Bits b0, b1, h0, h1, h2 and the carry's padded slot, at position 0.
    for assignment in 0..64u32 {
        let bit = |j: u32| i64::from((assignment >> j) & 1);
        let low: Vec<i64> = (0..2).map(bit).collect();
        let high: Vec<i64> = (2..5).map(bit).collect();
        let slot = bit(5);
        let edits_b: Edits = (0..2).map(|j| (vec![0, j], low[j as usize])).collect();
        let mut edits_h: Edits = (0..3).map(|j| (vec![0, j], high[j as usize])).collect();
        edits_h.push((vec![0, 3], slot));
        let verdict = verdict_of(&run, &[(b, edits_b), (h, edits_h)]);
        let honest = low == honest_b && high == honest_h && slot == 0;
        assert_eq!(verdict.is_ok(), honest, "assignment {assignment:06b}");
        if let Err(reject) = verdict {
            assert_eq!(reject, run.stage_1(), "assignment {assignment:06b}");
        }
        accepted += usize::from(verdict.is_ok());
    }
    assert_eq!(accepted, 1, "one passing witness");
    // Non-Boolean bits that recompose 13: b = (3, −1), and h = (3, 0, 0) for 3.
    for (name, b_values, h_values) in [
        ("b = (3, −1)", vec![3, -1], honest_h.clone()),
        ("h = (3, 0, 0)", honest_b.clone(), vec![3, 0, 0]),
        ("b = (−3, 0), h = (0, 0, 1)", vec![-3, 0], vec![0, 0, 1]),
    ] {
        let edits_b: Edits = (0..2u32)
            .map(|j| (vec![0, j], b_values[j as usize]))
            .collect();
        let edits_h: Edits = (0..3u32)
            .map(|j| (vec![0, j], h_values[j as usize]))
            .collect();
        let recomposed: i64 =
            b_values[0] + 2 * b_values[1] + 4 * (h_values[0] + 2 * h_values[1] + 4 * h_values[2]);
        assert_eq!(recomposed, 13, "{name} recomposes 13");
        assert_eq!(
            verdict_of(&run, &[(b, edits_b), (h, edits_h)]),
            Err(run.stage_1()),
            "{name}"
        );
    }
}

/// Values at positions of a tensor: each position's indices, outermost first, and its value.
type Edits = Vec<(Vec<u32>, i64)>;

/// The verdict on a prover that commits the honest witness with `edits`, each a tensor and its
/// positions' values.
fn verdict_of(run: &Bits, edits: &[(TensorId, Edits)]) -> Result<Vec<F>, Reject> {
    let tables = edits
        .iter()
        .map(|(tensor, at)| {
            let at: Vec<(&[u32], i64)> = at.iter().map(|(p, v)| (p.as_slice(), *v)).collect();
            run.edited(*tensor, &at)
        })
        .collect();
    let cheat = Cheat {
        witness: &run.witness,
        tables,
    };
    run.verify(
        &run.schedule,
        &run.prove(&run.schedule, &cheat),
        &run.public,
    )
}

/// RFC-0066 '`AssertEqual`': against a public side, only the committed side equal to it passes;
/// one off by `δ` for `δ ∈ {−2, −1, 1, 2}` at any position, its bits recomputed so that its own
/// decomposition holds, is rejected at stage 1's final check.
#[test]
fn equal_tensors_are_the_unique_passing_pair() {
    let values = vec![7, 9, 250, 3, 128, 64];
    let mut g = GraphBuilder::new();
    let u = declared(&mut g, &[3, 2], ADVICE, U8.0, U8.1);
    let (bu, _) = decompose(&mut g, u, &[3, 2], 8, 0);
    let p = declared(&mut g, &[3, 2], PUBLIC, U8.0, U8.1);
    g.node(Op::AssertEqual, &[u, p], Vec::new())
        .expect("a binding");
    let run = Bits::of(g.finish(), vec![(u, values.clone()), (p, values.clone())]);
    for position in 0..6u32 {
        let at = [position / 2, position % 2];
        for delta in [-2, -1, 0, 1, 2] {
            let value = values[position as usize] + delta;
            let (bits, _) = bits_of(value, 8, 0);
            let mut edits_b: Edits = Vec::new();
            for (j, &bit) in (0u32..).zip(&bits) {
                edits_b.push((vec![at[0], at[1], j], bit));
            }
            let verdict = verdict_of(&run, &[(u, vec![(at.to_vec(), value)]), (bu, edits_b)]);
            if delta == 0 {
                assert!(verdict.is_ok(), "position {position}: the honest pair");
            } else {
                assert_eq!(
                    verdict,
                    Err(run.stage_1()),
                    "position {position}, δ = {delta}"
                );
            }
        }
    }
}

/// RFC-0072 'The bit zero-checks': `AssertEqual` compares real positions only. A committed side
/// whose padded row holds 7 and 9, against a public side padded with 0, is an honest witness of
/// RFC-0066's relation and verifies; a real position one off is still rejected. Every other
/// `AssertEqual` test has the two sides equal on the whole padded cube, so only this one pins the
/// selector `ψ̃`'s polarity, in the prover and in the verifier.
#[test]
fn assert_equal_ignores_padded_positions() {
    let values = vec![7, 0, 255, 3, 128, 64];
    let mut g = GraphBuilder::new();
    let v = declared(&mut g, &[3, 2], ADVICE, U8.0, U8.1);
    let p = declared(&mut g, &[3, 2], PUBLIC, U8.0, U8.1);
    g.node(Op::AssertEqual, &[v, p], Vec::new())
        .expect("an equality");
    let run = Bits::of(g.finish(), vec![(v, values.clone()), (p, values)]);
    let padded = verdict_of(&run, &[(v, vec![(vec![3, 0], 7), (vec![3, 1], 9)])]);
    assert!(
        padded.is_ok(),
        "padded positions are not compared: {padded:?}"
    );
    let real = verdict_of(&run, &[(v, vec![(vec![2, 1], 65)])]);
    assert_eq!(real, Err(run.stage_1()), "a real position one off");
}

/// RFC-0072 'Recomposition': a bit set at a padded row of `x`, where `x` is honestly 0, fails
/// recomposition alone, with no input padding check involved; a non-Boolean pair there that
/// recomposes 0 fails booleanity. Three ways for `x` to be 0 there: a committed input's padded
/// row (pinned), a public input's padded row (the verifier's 0), and the padded column of an
/// `Expand` of a committed input that has no padded position, so no input padding check.
#[test]
fn a_bit_at_a_padded_row_of_x_fails_recomposition() {
    let (run, _, b) = bytes(&[17, 255, 0, 128, 2, 99]);
    let one_bit = run.edited(b, &[(&[3, 1, 0], 1)]);
    let pair = run.edited(b, &[(&[3, 1, 0], 2), (&[3, 1, 1], -1)]);
    run.caught(&[
        (
            "a committed input's padded row: a bit",
            vec![one_bit],
            (Relation::Recomposition, b),
        ),
        (
            "a committed input's padded row: a non-Boolean pair",
            vec![pair],
            (Relation::Booleanity, b),
        ),
    ]);

    let mut g = GraphBuilder::new();
    let p = declared(&mut g, &[3], PUBLIC, U8.0, U8.1);
    let (b, _) = decompose(&mut g, p, &[3], 8, 0);
    let run = Bits::of(g.finish(), vec![(p, vec![1, 2, 3])]);
    let bit = run.edited(b, &[(&[3, 5], 1)]);
    run.caught(&[(
        "a public input's padded row: a bit",
        vec![bit],
        (Relation::Recomposition, b),
    )]);

    let mut g = GraphBuilder::new();
    let u = declared(&mut g, &[4], ADVICE, U8.0, U8.1);
    let x = one(&mut g, Op::Expand { axis: 1, len: 3 }, &[u], {
        let mut d = decl(&[4, 3], NodeKind::Virtual);
        d.range = IntRange { lo: U8.0, hi: U8.1 };
        d
    });
    let (b, _) = decompose(&mut g, x, &[4, 3], 8, 0);
    let run = Bits::of(g.finish(), vec![(u, vec![9, 200, 0, 77])]);
    assert!(
        !relations(&run)
            .iter()
            .any(|(r, _)| *r == Relation::InputPadding),
        "u has no padded position"
    );
    let bit = run.edited(b, &[(&[1, 3, 2], 1)]);
    run.caught(&[(
        "an Expand's padded column: a bit",
        vec![bit],
        (Relation::Recomposition, b),
    )]);
}

/// RFC-0066 Lemma 1 at Goldilocks' boundary (`soundness-review` 'Wrap attacks'): the widest
/// decomposition, `{61, 1}`, recomposes at most `2^62 − 1 < P`, so no Boolean bits recompose a
/// committed `u = −1 = P − 1` or `u = 2^62`. With every bit 1 (`2^62 − 1`) both are rejected at
/// stage 1, while an honest `u = 2^62 − 1` verifies.
#[test]
fn the_widest_decomposition_does_not_wrap() {
    let mut g = GraphBuilder::new();
    let top = (1i64 << 62) - 1;
    let u = declared(&mut g, &[2], ADVICE, 0, i128::from(top));
    let (b, h) = decompose(&mut g, u, &[2], 61, 1);
    let h = h.expect("one high bit");
    let run = Bits::of(g.finish(), vec![(u, vec![top, 12_345])]);
    assert!(verdict_of(&run, &[]).is_ok(), "an honest 2^62 − 1");
    let ones_b: Edits = (0..61u32).map(|j| (vec![1, j], 1)).collect();
    let ones_h: Edits = vec![(vec![1, 0], 1)];
    for (name, value) in [("u = −1", -1i64), ("u = 2^62", 1i64 << 62)] {
        let verdict = verdict_of(
            &run,
            &[
                (u, vec![(vec![1], value)]),
                (b, ones_b.clone()),
                (h, ones_h.clone()),
            ],
        );
        assert_eq!(verdict, Err(run.stage_1()), "{name}");
    }
}

// ----- The identity and its closed forms ---------------------------------------------------------

/// RFC-0072 'Recomposition' 'Why it holds': for each `y`, the instance's sum over the bit axis,
/// `Σ_{i ∈ {0,1}^k} [2^{−k}·x − W̃_b(i)·b̂(i) − 2^{low}·W̃_h(i)·ĥ(i)]`, is RFC-0066's relation
/// `x − Σ_{j < low} 2^j·b_j − 2^{low}·Σ_{j < high} 2^j·h_j`, for any values of `x`, the bits and
/// the padded slots, with the verifier's closed forms at the cube's points and the outputs
/// zero-extended on their high coordinates: `(8, 0)`, `(32, 3)`, `(2, 8)` with `k_h > k_b`, and
/// `(3, 5)`. And `W̃`'s closed form, both of its forms, is the extension of the weights' table at
/// random points.
#[test]
fn recomposition_summed_over_the_bit_axis_is_rfc_0066s_relation() {
    let mut rng = random_points(1, 4096, 0x0526_2100)
        .into_iter()
        .map(|p| p[0]);
    for (low, high) in [(8u8, 0u8), (32, 3), (2, 8), (3, 5)] {
        let bit_vars = |bits: u8| axis_vars(u32::from(bits)) as usize;
        let (k_b, k_h) = (bit_vars(low), if high > 0 { bit_vars(high) } else { 0 });
        let k = k_b.max(k_h);
        // Any values, the padded slots' among them: the identity holds for every table.
        let x = rng.next().expect("a value");
        let b: Vec<F> = (0..1usize << k_b)
            .map(|_| rng.next().expect("a value"))
            .collect();
        let h: Vec<F> = (0..1usize << k_h)
            .map(|_| rng.next().expect("a value"))
            .collect();
        let cube = |i: usize| -> Vec<F> {
            (0..k)
                .map(|t| if (i >> t) & 1 == 1 { F::ONE } else { F::ZERO })
                .collect()
        };
        let mut sum = F::ZERO;
        for i in 0..1usize << k {
            let point = cube(i);
            // `b̂(i) = Z_b(i)·b̃(i_{≥ k − k_b})`, at a vertex the table's entry `i >> (k − k_b)`.
            let b_hat = zero_extension(&point, k - k_b) * b[i >> (k - k_b)];
            let mut term =
                two_to(k as u64).inverse().expect("2 ≠ 0") * x - weights_at(&point, low) * b_hat;
            if high > 0 {
                let h_hat = zero_extension(&point, k - k_h) * h[i >> (k - k_h)];
                term -= two_to(u64::from(low)) * weights_at(&point, high) * h_hat;
            }
            sum += term;
        }
        let mut relation = x;
        for (j, &bit) in (0u64..).zip(&b).take(usize::from(low)) {
            relation -= two_to(j) * bit;
        }
        for (j, &bit) in (0u64..).zip(&h).take(usize::from(high)) {
            relation -= two_to(u64::from(low) + j) * bit;
        }
        assert_eq!(sum, relation, "({low}, {high})");
        // `W̃`'s closed form against the extension of the weights' table on `k` variables.
        for (bits, k_o) in [(low, k_b), (high, k_h)] {
            if bits == 0 {
                continue;
            }
            let pad = k - k_o;
            let table: Vec<Goldilocks> = (0..1usize << k)
                .map(|i| {
                    let j = i >> pad;
                    if i & ((1 << pad) - 1) == 0 && j < usize::from(bits) {
                        Goldilocks::from_u64_reduced(1 << j)
                    } else {
                        Goldilocks::ZERO
                    }
                })
                .collect();
            for r in random_points(k, 4, 0x0526_2200 + u64::from(bits)) {
                assert_eq!(
                    weights_at(&r, bits),
                    evaluate_base::<F>(&table, &r).expect("a point"),
                    "W̃ of {bits} bits on {k} variables"
                );
            }
        }
    }
}
