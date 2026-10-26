// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The SHA-256 circuit end to end (RFC-0072 'Testing Strategy' 'The circuits', 'Padded
//! positions', 'The bit zero-checks'; RFC-0066 'The SHA-256 circuit'; #526's slice 3b), built by
//! `tessera_compiler::anchor`, its ranges settled by the range analysis, run on the CPU engine,
//! proven by the stage prover and checked by the verifier of this test, every public factor its
//! own: the bit zero-checks' closed forms in [`super::bits`], the selections' and `BitFn`'s in
//! [`super::selections`].
//!
//! - **The key commitment**, one block of `"tessera-anchor-key/v1" ‖ k_s ‖ salt_k` with `k_s` and
//!   `salt_k` committed and the digest bound, proves and verifies in 14 stages, the report's
//!   count, its pool the committed tensors' extensions.
//! - **Padded positions.** Three key commitments, `M = 3` padded to 4, and two blocks of three
//!   public messages prove and verify.
//! - **Rejections**, each at stage 1's final check, `Reject(Sumcheck { stage: 1, round: n_1 })`: a
//!   digest one bit off, the verifier's public digest that is not the hash; and a round-0 carry
//!   bit off, committed by the prover under the honest virtual tensors. Each is accepted with
//!   exactly the one stage-1 instance it breaks removed ([`bits::without`]), which the test finds
//!   by removing each candidate in turn, the pool then the cheat's own committed tables'.

use super::bits::{self, without};
use super::selections::run_with;
use super::*;
use tessera_compiler::anchor::{
    SHA256_IV, SHA256_MASKS, Sha256, Sha256Layout, sha256, sha256_bind, sha256_bits,
    sha256_digest_input, sha256_inputs, sha256_pad,
};
use tessera_compiler::{analyze, settle};

/// RFC-0059's key commitment label.
const LABEL: &[u8] = b"tessera-anchor-key/v1";

/// A circuit: its graph, its inputs in `sha256_inputs`'s order, its tensors, and its digest
/// input when bound.
struct Circuit {
    graph: Graph,
    inputs: Vec<TensorId>,
    sha: Sha256,
    digest: Option<TensorId>,
}

/// The circuit of `layout`, bound to a digest input when `bind`, its ranges the analysis's.
fn circuit(layout: &Sha256Layout, bind: bool) -> Circuit {
    let mut g = GraphBuilder::new();
    let inputs: Vec<TensorId> = sha256_inputs(layout)
        .into_iter()
        .map(|d| g.input(d).expect("an input"))
        .collect();
    let sha = sha256(&mut g, layout, &inputs).expect("the circuit");
    let digest = bind.then(|| {
        let digest = g
            .input(sha256_digest_input(layout.messages))
            .expect("the digest");
        let last = sha.chaining.last().expect("a block");
        sha256_bind(&mut g, last, digest, layout.messages).expect("the binding");
        digest
    });
    let mut graph = g.finish();
    let ranges = analyze(&graph, &[]).expect("the circuit's ranges");
    settle(&mut graph, &ranges).expect("bounded ranges");
    Circuit {
        graph,
        inputs,
        sha,
        digest,
    }
}

/// The inputs' values on the padded messages `padded`: the public bits, zero at committed
/// bytes, the IV's, the masks, and when bytes are committed, the bytes `[K, M]` and `place`.
/// Test data, so plain buffers (a prover reads committed bytes from its secret-class ones).
fn values(
    layout: &Sha256Layout,
    circuit: &Circuit,
    padded: &[Vec<u8>],
) -> Vec<(TensorId, Vec<i64>)> {
    let rows: Vec<&[u8]> = padded.iter().map(Vec::as_slice).collect();
    let iv: Vec<u8> = SHA256_IV.iter().flat_map(|w| w.to_be_bytes()).collect();
    let ivs = vec![iv.as_slice(); padded.len()];
    let mut values = vec![
        (circuit.inputs[0], layout.public_bits(&rows)),
        (circuit.inputs[1], sha256_bits(&ivs)),
        (
            circuit.inputs[2],
            SHA256_MASKS.iter().flatten().copied().collect(),
        ),
    ];
    if !layout.committed.is_empty() {
        let bytes = layout
            .committed
            .iter()
            .flat_map(|&p| rows.iter().map(move |m| i64::from(m[p as usize])))
            .collect();
        values.push((circuit.inputs[3], bytes));
        values.push((circuit.inputs[4], layout.place()));
    }
    values
}

/// The digest bits `[256, M]` the circuit computes on `padded`: its last block's chaining words,
/// word `j` on rows `32j` to `32j + 31`, from the unbound circuit run on the engine.
fn digest(layout: &Sha256Layout, padded: &[Vec<u8>]) -> Vec<i64> {
    let unbound = circuit(layout, false);
    let inputs = values(layout, &unbound, padded)
        .into_iter()
        .map(|(t, v)| (t, v.into_iter().collect()))
        .collect();
    let witness = Engine::for_graph(&unbound.graph, &[])
        .expect("an engine")
        .evaluate(inputs)
        .expect("the digest");
    let last = unbound.sha.chaining.last().expect("a block");
    last.iter()
        .flat_map(|&w| witness.tensor(w).expect("a word").to_vec())
        .collect()
}

/// Key commitments of `m` sessions, each `"tessera-anchor-key/v1" ‖ k_s ‖ salt_k` of 53 bytes,
/// padded to one block, with their 32 key and salt bytes committed.
fn key_commitments(m: u32, seed: u64) -> (Sha256Layout, Vec<Vec<u8>>) {
    let mut state = seed;
    let mut byte = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        u8::try_from(state >> 56).expect("a byte")
    };
    let padded = (0..m)
        .map(|_| {
            let mut message = LABEL.to_vec();
            message.extend((0..32).map(|_| byte()));
            sha256_pad(&message)
        })
        .collect();
    let layout = Sha256Layout {
        blocks: 1,
        messages: m,
        committed: (21..53).collect(),
    };
    (layout, padded)
}

/// The honest run of `layout`'s bound circuit on `padded`, its digest the hash: its stage count
/// the report's and `expected`, and the proof accepted, its pool the prover's and the committed
/// tensors' extensions.
fn proven(layout: &Sha256Layout, padded: &[Vec<u8>], expected: usize) -> (Circuit, Run) {
    let bound = circuit(layout, true);
    let mut inputs = values(layout, &bound, padded);
    inputs.push((bound.digest.expect("bound"), digest(layout, padded)));
    let run = run_with(&bound.graph, inputs, 0x0526_3B00);
    assert_eq!(
        run.schedule.stage_count(),
        expected,
        "RFC-0072 'Chain length'"
    );
    let pool = verify(&bound, &run, &run.schedule, &run.proof, &run.public)
        .expect("the honest proof verifies");
    assert_eq!(pool, run.pool, "the prover's pool");
    let coords = coords_after(&run.schedule, &run.proof, &run.points);
    assert_eq!(
        pool,
        pool_of_graph(&coords, &bound.graph, &run.schedule, &run.witness),
        "the pool is the committed tensors' extensions"
    );
    (bound, run)
}

/// The verifier's public data: the public inputs of the statement it holds, the digest's word
/// slices computed from its digest, and every other public tensor, a function of public inputs
/// no test here changes, from the honest run. (Running the whole graph on the public inputs
/// alone, as [`public_values`] does, would fail the binding's `AssertEqual` when bytes are
/// committed.)
struct Statement<'a> {
    inputs: &'a [(TensorId, Vec<i64>)],
    slices: Vec<(TensorId, Vec<i64>)>,
    witness: &'a Witness,
    public: Vec<bool>,
}

impl WitnessSource for Statement<'_> {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        if let Some((_, values)) = self
            .inputs
            .iter()
            .chain(&self.slices)
            .find(|(t, _)| *t == tensor)
        {
            return Some(values);
        }
        if !self.public.get(tensor.0 as usize).copied().unwrap_or(false) {
            return None;
        }
        self.witness.values(tensor)
    }
}

/// The verifier's verdict on `proof` of `bound`'s circuit under `schedule` when it holds the
/// public inputs `statement`.
fn verify(
    bound: &Circuit,
    run: &Run,
    schedule: &StageSchedule,
    proof: &Proof,
    statement: &[(TensorId, Vec<i64>)],
) -> Result<Vec<F>, Reject> {
    let digest = bound.digest.expect("bound");
    let bits = &statement
        .iter()
        .find(|(t, _)| *t == digest)
        .expect("the digest")
        .1;
    let m = bound.graph.tensors[digest.0 as usize].shape[1] as usize;
    let slices = bound
        .graph
        .nodes
        .iter()
        .filter(|n| n.inputs == [digest])
        .map(|n| {
            let Op::Slice { start, len, .. } = n.op else {
                panic!("a digest word's slice")
            };
            let (from, to) = (start as usize * m, (start + len) as usize * m);
            (n.outputs[0], bits[from..to].to_vec())
        })
        .collect();
    let values = Statement {
        inputs: statement,
        slices,
        witness: &run.witness,
        public: public_tensors(&bound.graph),
    };
    let publics = Publics {
        values: &values,
        shift: Shift::Extension,
        kernel: Kernel::Closed,
    };
    verify_public(&bound.graph, &[], schedule, proof, &run.points, publics)
}

/// RFC-0072 'Testing Strategy' 'The circuits': the SHA-256 key commitment, one block with `k_s`
/// and `salt_k` committed, proves on the CPU prover and verifies, in 14 stages.
#[test]
fn the_key_commitment_proves_and_verifies() {
    let (layout, padded) = key_commitments(1, 0x0526_3B01);
    let (_, run) = proven(&layout, &padded, 14);
    let instances: usize = run.schedule.stages.iter().map(|s| s.instances.len()).sum();
    assert_eq!(instances, 3431, "RFC-0072's sizes table");
}

/// RFC-0072 'Padded positions': three key commitments, `M = 3` padded to 4, prove and verify,
/// and so do three public messages of two blocks, whose chaining runs inside the graph.
#[test]
fn sha256_of_three_messages_proves_and_verifies() {
    let (layout, padded) = key_commitments(3, 0x0526_3B02);
    proven(&layout, &padded, 14);
    // Messages of 56 to 119 bytes take two blocks.
    let padded: Vec<Vec<u8>> = [(0x61u8, 60usize), (0x62, 100), (0x63, 56)]
        .iter()
        .map(|&(byte, len)| sha256_pad(&vec![byte; len]))
        .collect();
    assert!(padded.iter().all(|p| p.len() == 128));
    let layout = Sha256Layout {
        blocks: 2,
        messages: 3,
        committed: Vec::new(),
    };
    proven(&layout, &padded, 10);
}

/// Stage 1's instances of `relation` whose node is `node`, by index.
fn checks_of(run: &Run, node: tessera_ir::NodeId, relation: Relation) -> Vec<usize> {
    run.schedule.stages[1]
        .instances
        .iter()
        .enumerate()
        .filter(|(_, i)| i.node == NodeRef::Node(node) && i.relation == relation)
        .map(|(index, _)| index)
        .collect()
}

/// The instances of every stage of `schedule`.
fn instances(schedule: &StageSchedule) -> usize {
    schedule.stages.iter().map(|s| s.instances.len()).sum()
}

/// RFC-0072 'The bit zero-checks' 'Rejections': a digest one bit off, the statement's digest
/// differing from the hash in one bit of one message, fails its word's `AssertEqual` at stage
/// 1's final check, and is accepted with exactly that check removed.
///
/// The changed digest reaches exactly one instance: the digest is read only by the binding's
/// eight word slices, a slice only by its word's `AssertEqual`, whose public side it is, and
/// nothing else reads a public tensor's value. That instance is found from the graph, the one
/// whose second input slices rows `32j` to `32j + 31` of the digest, the flipped row among them.
/// With it removed (its one claim, on the computed word's committed bits, leaves the pool with
/// it), the honest prover's proof under that schedule is accepted, its pool the committed
/// tensors' extensions; with another word's removed in its place, it is still rejected.
#[test]
fn a_digest_one_bit_off_is_rejected_at_stage_1() {
    let (layout, padded) = key_commitments(3, 0x0526_3B03);
    let (bound, run) = proven(&layout, &padded, 14);
    let digest = bound.digest.expect("bound");
    let m = layout.messages as usize;
    let off = |row: usize, message: usize| {
        let mut statement = run.public.clone();
        let (_, bits) = statement
            .iter_mut()
            .find(|(t, _)| *t == digest)
            .expect("the digest");
        bits[row * m + message] ^= 1;
        statement
    };
    let cases = [(0usize, 0usize), (77, 2), (255, 1)];
    for (row, message) in cases {
        assert_eq!(
            verify(&bound, &run, &run.schedule, &run.proof, &off(row, message)),
            Err(run.end(1)),
            "digest row {row} of message {message} one bit off"
        );
    }
    // Each word's `AssertEqual` instance, by the digest rows its public side slices.
    let words: Vec<(core::ops::Range<usize>, usize)> = bound
        .graph
        .nodes
        .iter()
        .filter(|n| n.op == Op::AssertEqual)
        .map(|n| {
            let slice = bound
                .graph
                .nodes
                .iter()
                .find(|s| s.outputs == [n.inputs[1]] && s.inputs == [digest])
                .expect("the word's slice of the digest");
            let Op::Slice { start, len, .. } = slice.op else {
                panic!("a slice")
            };
            let [check] = checks_of(&run, n.id, Relation::AssertEqual)[..] else {
                panic!("one AssertEqual instance per word")
            };
            (start as usize..(start + len) as usize, check)
        })
        .collect();
    assert_eq!(words.len(), 8);
    for (row, message) in cases {
        let name = format!("digest row {row} of message {message} one bit off");
        let statement = off(row, message);
        let [(_, check)] = words
            .iter()
            .filter(|(rows, _)| rows.contains(&row))
            .cloned()
            .collect::<Vec<_>>()[..]
        else {
            panic!("one word holds row {row}")
        };
        let reduced = without(&run.schedule, check);
        assert_eq!(instances(&reduced), instances(&run.schedule) - 1, "{name}");
        let proof = prove_graph(&bound.graph, &[], &reduced, &run.witness, &run.points).0;
        let pool = verify(&bound, &run, &reduced, &proof, &statement)
            .unwrap_or_else(|reject| panic!("{name}: accepted without its check, not {reject:?}"));
        let coords = coords_after(&reduced, &proof, &run.points);
        assert_eq!(
            pool,
            pool_of_graph(&coords, &bound.graph, &reduced, &run.witness),
            "{name}: the openings accept the committed tensors"
        );
        // Another word's check removed in its place: still rejected.
        let (_, other) = words
            .iter()
            .find(|(rows, _)| !rows.contains(&row))
            .expect("another word");
        let reduced = without(&run.schedule, *other);
        let proof = prove_graph(&bound.graph, &[], &reduced, &run.witness, &run.points).0;
        assert_eq!(
            verify(&bound, &run, &reduced, &proof, &statement),
            Err(Reject {
                stage: 1,
                round: reduced.stages[1].rounds
            }),
            "{name}: another word's check removed"
        );
    }
}

/// RFC-0072 'Testing Strategy' 'No claim unchecked', 'The bit zero-checks': a round-0 carry bit
/// off in the key commitment, the prover committing round 0's `e` sum's carry `h` with one bit
/// flipped under the honest sums, fails its recomposition at stage 1's final check, and is
/// accepted with exactly that check removed.
///
/// The cheat changes one real slot of `h` and no other table, so only the instances that read
/// `h` can see it: its node's booleanity of `h`, which a flipped bit still passes, its bit
/// padding, which reads the padded fourth slot alone, and its recomposition. The test finds the
/// one that breaks rather than assume it: with each of the node's four stage-1 instances removed
/// in turn (booleanity of `b` and of `h`, bit padding of `h`, recomposition), only the
/// recomposition's removal is accepted.
///
/// Removing the recomposition removes what proves its claims and nothing else (`bits::without`):
/// its claims on `b` and `h` leave the pool, and its claim on `x`, the sum's virtual value,
/// takes with it the instances of later stages that prove that claim alone. They are the
/// value's five combined selections (stages 2 to 6), then its rows' one part that is neither
/// public nor shared, the message's first word, through its slice, the `Xor`, the placing `Mul`
/// and the assembly's gather (stages 7 to 10), whose claim on the committed bits is the third
/// pool claim gone: ten instances and three pool claims in all. Each is a relation of honest
/// tables that no other claim reads. Accepted, the pool is the cheat's own committed tables'
/// extensions, the flipped carry among them, through its booleanity's claim.
#[test]
fn a_round_0_carry_off_is_rejected_at_stage_1() {
    let (layout, padded) = key_commitments(1, 0x0526_3B04);
    let (bound, run) = proven(&layout, &padded, 14);
    // The schedule's sums carry 2 bits and come first; round 0's `e` sum is the first of 3.
    let node = bound
        .graph
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::BitDecompose { low: 32, high: 3 }))
        .expect("round 0's e sum");
    let carry = node.outputs[1];
    let shape = bound.graph.tensors[carry.0 as usize].shape.clone();
    let cheat = |bit: u32| {
        let mut table = bits::table_of(&run.witness, &bound.graph, carry);
        table[bits::index(&shape, &[0, bit])] ^= 1;
        bits::Cheat {
            witness: &run.witness,
            tables: vec![(carry, table)],
        }
    };
    for bit in 0..3 {
        let bad = prove_graph(&bound.graph, &[], &run.schedule, &cheat(bit), &run.points).0;
        assert_eq!(
            verify(&bound, &run, &run.schedule, &bad, &run.public),
            Err(run.end(1)),
            "carry bit {bit}"
        );
    }
    let [b, h] = node.outputs[..] else {
        panic!("a sum's word and carry")
    };
    let kind = |t: TensorId| run.schedule.layers.outside(t).0;
    let on = |relation: Relation, tensor: TensorId| {
        checks_of(&run, node.id, relation)
            .into_iter()
            .filter(|&i| run.schedule.stages[1].instances[i].kind == kind(tensor))
            .collect::<Vec<_>>()
    };
    let [recomposition] = on(Relation::Recomposition, b)[..] else {
        panic!("one recomposition, on b's kind")
    };
    let others = [
        ("booleanity of b", on(Relation::Booleanity, b)),
        ("booleanity of h", on(Relation::Booleanity, h)),
        ("bit padding of h", on(Relation::BitPadding, h)),
    ];
    assert_eq!(
        checks_of(&run, node.id, Relation::Booleanity).len()
            + checks_of(&run, node.id, Relation::BitPadding).len()
            + 1,
        4,
        "the node's four stage-1 instances"
    );
    for bit in 0..3 {
        let cheat = cheat(bit);
        for (name, check) in &others {
            let [check] = check[..] else {
                panic!("one {name}")
            };
            let reduced = without(&run.schedule, check);
            let bad = prove_graph(&bound.graph, &[], &reduced, &cheat, &run.points).0;
            assert_eq!(
                verify(&bound, &run, &reduced, &bad, &run.public),
                Err(Reject {
                    stage: 1,
                    round: reduced.stages[1].rounds
                }),
                "carry bit {bit} with the {name} removed"
            );
        }
        let reduced = without(&run.schedule, recomposition);
        assert_eq!(reduced.stage_count(), run.schedule.stage_count());
        let gone: Vec<usize> = run
            .schedule
            .stages
            .iter()
            .zip(&reduced.stages)
            .map(|(before, after)| before.instances.len() - after.instances.len())
            .collect();
        assert_eq!(
            gone,
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            "the recomposition, and the chain proving the sum's value"
        );
        assert_eq!(run.schedule.pool.len() - reduced.pool.len(), 3);
        let bad = prove_graph(&bound.graph, &[], &reduced, &cheat, &run.points).0;
        let pool = verify(&bound, &run, &reduced, &bad, &run.public).unwrap_or_else(|reject| {
            panic!("carry bit {bit}: accepted without its recomposition, not {reject:?}")
        });
        let coords = coords_after(&reduced, &bad, &run.points);
        let opened = pool_of_tables(
            &coords,
            &bound.graph,
            &reduced,
            &|t| match cheat.padded(t) {
                Some(values) => values.iter().map(|&v| Goldilocks::from_i64(v)).collect(),
                None => super::padded(&run.witness, &bound.graph, t),
            },
        );
        assert_eq!(
            pool, opened,
            "carry bit {bit}: the openings accept the cheat's tables"
        );
    }
}
