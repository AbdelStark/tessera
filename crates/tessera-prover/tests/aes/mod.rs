// SPDX-License-Identifier: Apache-2.0 OR MIT

//! AES-128-CTR end to end (RFC-0072 'Testing Strategy' 'The circuits', 'AES's graphs', 'Padded
//! positions', 'No claim unchecked'; RFC-0015 'Circuits' step 2; #526's fourth slice), built by
//! `tessera_compiler::anchor`, its key and plaintext range-checked by a `BitDecompose { 8, 0 }` of
//! each, since alone the circuit is a fragment (RFC-0006). It runs on the CPU engine, the leaf
//! stage proves its 127 lookups through `tessera-logup` and the split, the stage prover its
//! stages, and the verifier of [`super::lookups`] checks all of it, the trees with the production
//! driver (`tv_core::leaf`, `tv-tables`), on its own transcript.
//!
//! - **One block under a committed key**, NIST SP 800-38A's CTR-AES128 vector, proves and
//!   verifies in 3 stages, the report's count, its openings the committed tables' extensions.
//! - **`B = 3` blocks**, padded to 4, prove and verify. The padded block runs on a zero counter
//!   under zero round keys: its S-box outputs are `AesSbox(0) = 0x63` and its consequences,
//!   key-independent, committed with the block's other lookup outputs. The plaintext's input
//!   padding check pins its padded block to 0.
//! - **RFC-0072's sizes row**: 97 selections, 30 `ConstMul`s per channel, 127 lookups, 154
//!   instances, 154 evaluations, 332 split values, 3 stages, at both block counts, with the
//!   range checks' zero-checks beside them.
//! - **Rejections.** An S-box output off its table, and the padded block's S-box outputs taken as
//!   0, are `Reject(Lookup { table })` of the S-box; a counter block changed on the verifier's side
//!   is `Reject(Sumcheck { stage: 0, round: 0 })`; and `Gf256Mul { 3 }`'s lookup run on another
//!   column than `shifted`'s, its leaf value that column's, is `Reject(Sumcheck { stage: 1, round:
//!   n_1 })`, which a verifier merging the two lookups' claims on `shifted` accepts.

use super::lookups::{
    Claims, Committing, Override, Rejection, Setup, Verdict, committed, multiplicities, opened,
    opened_under, targets_are_the_verifiers, verify,
};
use super::padding::PublicOf;
use super::*;
use tessera_compiler::anchor::{AesCtr, AesTables, RCON, aes128_ctr, aes128_ctr_inputs};
use tessera_compiler::stage_count;
use tessera_prover::leaf::GraphLeafWitness;

const TABLES: AesTables = AesTables {
    sbox: TableId(0),
    mul2: TableId(1),
    mul3: TableId(2),
    xor: TableId(3),
};

/// The circuit's four tables.
fn specs() -> Vec<TableSpec> {
    [
        (TABLES.sbox, TableKind::AesSbox),
        (TABLES.mul2, TableKind::Gf256Mul { c: 2 }),
        (TABLES.mul3, TableKind::Gf256Mul { c: 3 }),
        (
            TABLES.xor,
            TableKind::Bitwise8 {
                op: tessera_tables::BitOp::Xor,
            },
        ),
    ]
    .into_iter()
    .map(|(id, kind)| {
        GeneratedTable::new(id, kind)
            .expect("a table")
            .spec()
            .clone()
    })
    .collect()
}

/// The circuit of `blocks` counter blocks, its key and plaintext range-checked, ready to prove;
/// its inputs in `aes128_ctr_inputs`'s order and its tensors.
struct Circuit {
    setup: Setup,
    inputs: [TensorId; 4],
    ctr: AesCtr,
}

fn circuit(blocks: u32) -> Circuit {
    let mut g = GraphBuilder::new();
    let inputs = aes128_ctr_inputs(blocks).map(|d| g.input(d).expect("an input"));
    let [key, _, _, plaintext] = inputs;
    for (x, shape) in [(key, vec![16, 8]), (plaintext, vec![16, blocks, 8])] {
        let mut bits = decl(&shape, ADVICE);
        bits.range = IntRange { lo: 0, hi: 1 };
        g.node(Op::BitDecompose { low: 8, high: 0 }, &[x], vec![bits])
            .expect("a range check");
    }
    let ctr = aes128_ctr(&mut g, &TABLES, inputs, blocks).expect("the circuit");
    Circuit {
        setup: Setup::new(g.finish(), specs()),
        inputs,
        ctr,
    }
}

fn hex(s: &str) -> Vec<i64> {
    (0..s.len() / 2)
        .map(|i| i64::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// NIST SP 800-38A F.5.1, CTR-AES128.Encrypt: the key, three counter blocks, plaintext and
/// ciphertext blocks.
const KEY: &str = "2b7e151628aed2a6abf7158809cf4f3c";
const COUNTERS: [&str; 3] = [
    "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
    "f0f1f2f3f4f5f6f7f8f9fafbfcfdff00",
    "f0f1f2f3f4f5f6f7f8f9fafbfcfdff01",
];
const PLAINTEXT: [&str; 3] = [
    "6bc1bee22e409f96e93d7e117393172a",
    "ae2d8a571e03ac9c9eb76fac45af8e51",
    "30c81c46a35ce411e5fbc1191a0a52ef",
];
const CIPHERTEXT: [&str; 3] = [
    "874d6191b620e3261bef6864990db6ce",
    "9806f66b7970fdff8617187bb9fffdff",
    "5ae4df3edbd5d35e5b4f09020db03eab",
];

/// Blocks of 16 bytes in the byte-major layout `[16, B]`: byte `j` of block `k` at `j·B + k`.
fn byte_major(blocks: &[&str]) -> Vec<i64> {
    let b = blocks.len();
    let mut out = vec![0; 16 * b];
    for (k, block) in blocks.iter().enumerate() {
        for (j, byte) in hex(block).into_iter().enumerate() {
            out[j * b + k] = byte;
        }
    }
    out
}

/// The inputs of the first `blocks` blocks of the vector under `key`.
fn inputs(c: &Circuit, key: &str, blocks: usize) -> Vec<(TensorId, Vec<i64>)> {
    let [k, rcon, counters, plaintext] = c.inputs;
    vec![
        (k, hex(key)),
        (rcon, RCON.iter().flatten().copied().collect()),
        (counters, byte_major(&COUNTERS[..blocks])),
        (plaintext, byte_major(&PLAINTEXT[..blocks])),
    ]
}

/// The verifier's public data on the honest witness: its public tensors, the round constants'
/// rows and the counter blocks, functions of the public inputs alone.
fn statement<'w>(c: &Circuit, witness: &'w Witness) -> PublicOf<'w> {
    PublicOf {
        witness,
        public: public_tensors(&c.setup.graph),
    }
}

/// The honest run of `blocks` blocks under `key`: the ciphertext the vector gives, the proof
/// accepted, its pool the prover's, its openings the committed tables' extensions, in 3 stages.
/// The witness and the verdict.
fn proven(c: &Circuit, key: &str, blocks: usize) -> (Witness, Verdict) {
    let setup = &c.setup;
    let witness = setup.witness(inputs(c, key, blocks));
    if key == KEY {
        assert_eq!(
            witness
                .tensor(c.ctr.ciphertext)
                .expect("the ciphertext")
                .to_vec(),
            byte_major(&CIPHERTEXT[..blocks]),
            "NIST SP 800-38A F.5.1"
        );
    }
    let honest = setup.honest(&witness);
    assert_eq!(setup.schedule.stage_count(), 3, "RFC-0072 'Chain length'");
    assert_eq!(stage_count(&setup.graph), 3, "the report's count");
    let public = statement(c, &witness);
    let verdict = verify(
        &setup.graph,
        &setup.specs,
        &setup.schedule,
        &public,
        &honest.proof,
        Claims::Own,
    )
    .expect("the honest proof verifies");
    assert_eq!(verdict.pool, honest.pool, "the prover's pool");
    opened(
        setup,
        &verdict,
        &honest.padded,
        &multiplicities(setup, &honest.leaf),
    );
    targets_are_the_verifiers(setup, &honest.targets, &verdict, &honest.padded);
    (witness, verdict)
}

/// RFC-0072 'Testing Strategy' 'The circuits': one AES-128-CTR block under a committed key,
/// NIST SP 800-38A's first block, proves on the CPU prover and verifies, through the real leaf
/// stage, in 3 stages.
#[test]
fn one_aes_block_under_a_committed_key_proves_and_verifies() {
    let c = circuit(1);
    proven(&c, KEY, 1);
}

/// RFC-0072 'Padded positions': three blocks, `B = 3` padded to 4, prove and verify. The padded
/// block's S-box outputs are `AesSbox(0) = 0x63` at the first round, its counter and round keys 0,
/// and every lookup output of the padded block is a key-independent constant: the same under two
/// keys, while the real blocks' differ. The plaintext's padded block is pinned by its input padding
/// check, one of stage 1's instances, and is 0; the ciphertext's is then the padded keystream.
#[test]
fn three_aes_blocks_padded_to_four_prove_and_verify() {
    let c = circuit(3);
    let setup = &c.setup;
    let padding = setup.schedule.stages[1]
        .instances
        .iter()
        .filter(|i| i.relation == Relation::InputPadding)
        .count();
    assert_eq!(padding, 1, "the plaintext's input padding check");
    let (witness, _) = proven(&c, KEY, 3);
    let (other, _) = proven(&c, "000102030405060708090a0b0c0d0e0f", 3);
    let a = PaddedWitness::new(&setup.graph, &setup.specs, &witness).expect("padded");
    let b = PaddedWitness::new(&setup.graph, &setup.specs, &other).expect("padded");
    // The first round's S-box: its padded block, column 3 of each byte's row, is 0x63.
    let first = setup
        .graph
        .nodes
        .iter()
        .find(|n| {
            n.op == Op::TableMap { table: TABLES.sbox }
                && n.outputs
                    .iter()
                    .all(|&t| setup.graph.tensors[t.0 as usize].shape == [16, 3])
        })
        .expect("the first round's S-box")
        .outputs[0];
    let sbox = a.padded(first).expect("the S-box's padded table");
    assert!((0..16).all(|byte| sbox[4 * byte + 3] == 0x63));
    let mut lookups = 0;
    for node in &setup.graph.nodes {
        let lookup = matches!(node.op, Op::TableMap { .. } | Op::Bitwise { .. });
        let output = node.outputs.first().copied();
        let Some(t) =
            output.filter(|&t| lookup && setup.graph.tensors[t.0 as usize].shape == [16, 3])
        else {
            continue;
        };
        lookups += 1;
        let (x, y) = (
            committed(&setup.graph, &a, t),
            committed(&setup.graph, &b, t),
        );
        let padded = |table: &[i64]| (0..16).map(|byte| table[4 * byte + 3]).collect::<Vec<_>>();
        assert_eq!(padded(&x), padded(&y), "a padded block is key-independent");
        assert_ne!(x, y, "a real block is not");
    }
    // The first AddRoundKey, seven a round for rounds 1 to 9, two for round 10, the ciphertext.
    assert_eq!(lookups, 1 + 9 * 7 + 2 + 1, "the blocks' lookups");
    let plaintext = committed(&setup.graph, &a, c.inputs[3]);
    assert!((0..16).all(|byte| plaintext[4 * byte + 3] == 0));
    let keystream = committed(&setup.graph, &a, c.ctr.keystream);
    let ciphertext = committed(&setup.graph, &a, c.ctr.ciphertext);
    assert!((0..16).all(|byte| ciphertext[4 * byte + 3] == keystream[4 * byte + 3]));
}

/// RFC-0072 'Messages, rejections and stages', the circuits' sizes, AES's row: 97 selections, 30
/// `ConstMul`s per channel, 127 lookups, 154 instances, 154 evaluations, 332 split values and 3
/// stages, at `B = 1` and `B = 3`. The range checks add their zero-checks beside them: each
/// decomposition's booleanity and recomposition, two instances and three evaluations, and at
/// `B = 3` the plaintext's input padding check. The split values are the leaf stage's: 205 lookup
/// inputs that are not public and 127 outputs; no claim is split. The independent count's pool
/// is the schedule's and the 127 outputs' leaf claims, which are opening targets here.
#[test]
fn aes_ctr_s_sizes_are_rfc_0072s_row() {
    for blocks in [1u32, 3] {
        let c = circuit(blocks);
        let setup = &c.setup;
        let name = format!("{blocks} blocks");
        let public = public_tensors(&setup.graph);
        let count = |pick: &dyn Fn(&Op) -> bool| {
            setup
                .graph
                .nodes
                .iter()
                .filter(|n| pick(&n.op) && n.outputs.iter().all(|t| !public[t.0 as usize]))
                .count()
        };
        let selections = count(&|op| matches!(op, Op::Slice { .. } | Op::GatherPublic { .. }));
        let per_channel = count(
            &|op| matches!(op, Op::ConstMul { c } if setup.graph.constants[c.0 as usize].len() > 1),
        );
        let lookups = count(&|op| matches!(op, Op::TableMap { .. } | Op::Bitwise { .. }));
        assert_eq!([selections, per_channel, lookups], [97, 30, 127], "{name}");
        let stages = &setup.schedule.stages;
        let zero_checks = stages[1]
            .instances
            .iter()
            .filter(|i| i.source == Source::ZeroCheck)
            .count();
        let range_checks = 4 + usize::from(blocks == 3);
        assert_eq!(zero_checks, range_checks, "{name}: the range checks'");
        let instances: usize = stages.iter().map(|s| s.instances.len()).sum();
        let evaluations: usize = stages
            .iter()
            .flat_map(|s| &s.instances)
            .map(|i| i.claims.len())
            .sum();
        let range_evaluations = 6 + usize::from(blocks == 3);
        assert_eq!(
            [instances - range_checks, evaluations - range_evaluations],
            [154, 154],
            "{name}"
        );
        let witness = setup.witness(inputs(&c, KEY, blocks as usize));
        let honest = setup.honest(&witness);
        assert_eq!(
            setup.layout.split_len(),
            205 + 127,
            "{name}: the leaf stage's"
        );
        let splits: usize = honest.proof.splits.len()
            + honest
                .proof
                .stages
                .iter()
                .map(|s| s.splits.len())
                .sum::<usize>();
        assert_eq!(splits, 332, "{name}: every split value");
        assert_eq!(setup.schedule.stage_count(), 3, "{name}");
        // The independent count's pool, 338 or 339 range-checked, holds the 127 lookup outputs'
        // leaf claims too: here they are opening targets of their own, beside the schedule's
        // pool and the four instances' multiplicities.
        let model_pool = 332 + 6 + usize::from(blocks == 3);
        assert_eq!(setup.schedule.pool.len() + 127, model_pool, "{name}");
        assert_eq!(
            honest.targets.advice().len(),
            setup.schedule.pool.len() + 127 + 4,
            "{name}: the opening targets"
        );
        assert!(honest.targets.weights().is_empty(), "{name}");
    }
}

/// RFC-0072 'Messages, rejections and stages' 'Rejections', 'Testing Strategy' 'The circuits':
/// an S-box output off its table is `Reject(Lookup { table })` of the S-box, at LogUp's roots,
/// in a real block, and in the padded block with its outputs taken as 0, as tables padded with
/// zeros would hold them. The rest of the witness is honest, so the S-box's instance is the one
/// that fails.
#[test]
fn an_s_box_output_off_its_table_is_rejected_at_logup() {
    let c = circuit(3);
    let setup = &c.setup;
    let witness = setup.witness(inputs(&c, KEY, 3));
    let padded = PaddedWitness::new(&setup.graph, &setup.specs, &witness).expect("padded");
    let public = statement(&c, &witness);
    let sboxes: Vec<TensorId> = setup
        .graph
        .nodes
        .iter()
        .filter(|n| n.op == Op::TableMap { table: TABLES.sbox })
        .map(|n| n.outputs[0])
        .filter(|&t| setup.graph.tensors[t.0 as usize].shape == [16, 3])
        .collect();
    assert_eq!(sboxes.len(), 10, "one per round");
    let fifth = committed(&setup.graph, &padded, sboxes[4]);
    let mut off = fifth.clone();
    off[4 * 7 + 1] ^= 0x10;
    let mut zero = committed(&setup.graph, &padded, sboxes[0]);
    for byte in 0..16 {
        zero[4 * byte + 3] = 0;
    }
    for (name, tensor, table) in [
        ("round 5, byte 7 of block 1", sboxes[4], off),
        ("round 1's padded block", sboxes[0], zero),
    ] {
        let source = Committing {
            inner: &padded,
            tables: vec![(tensor, table)],
        };
        let leaf =
            GraphLeafWitness::new(&setup.layout, &setup.graph, &setup.schedule.layers, &source)
                .expect("the leaf tensors");
        let (proof, _) = setup.prove(&leaf, &source);
        let verdict = verify(
            &setup.graph,
            &setup.specs,
            &setup.schedule,
            &public,
            &proof,
            Claims::Own,
        );
        assert_eq!(
            verdict.err(),
            Some(Rejection::Lookup { table: TABLES.sbox }),
            "{name}"
        );
    }
}

/// RFC-0072 'Public tensors' 'In the leaf stage': the counter blocks are public, and the split
/// sends no value for them; the verifier evaluates them itself. A counter byte changed on its
/// side, in any block, fails the split's identity, `Reject(Sumcheck { stage: 0, round: 0 })`.
#[test]
fn a_counter_block_changed_on_the_verifiers_side_is_rejected_at_stage_0() {
    let c = circuit(3);
    let setup = &c.setup;
    let witness = setup.witness(inputs(&c, KEY, 3));
    let honest = setup.honest(&witness);
    let public = statement(&c, &witness);
    let counters = byte_major(&COUNTERS);
    for entry in [0, 17, 47] {
        let mut changed = counters.clone();
        changed[entry] ^= 1;
        let overridden = Override {
            inner: &public,
            tensor: c.inputs[2],
            values: changed,
        };
        assert_eq!(
            verify(
                &setup.graph,
                &setup.specs,
                &setup.schedule,
                &overridden,
                &honest.proof,
                Claims::Own,
            )
            .err(),
            Some(Rejection::Sumcheck { stage: 0, round: 0 }),
            "counter byte {entry}"
        );
    }
}

/// RFC-0072 'Testing Strategy' 'No claim unchecked', 'Combined selections' 'No merging across
/// claims': round 1's `Gf256Mul { 3 }` lookup and its `Gf256Mul { 2 }` lookup read one tensor,
/// `shifted`, at one leaf point. A prover runs the `Gf256Mul { 3 }` lookup on another column, the
/// round's S-box output before `ShiftRows`, and sends that column's value at the leaf point as the
/// lookup's input: every lookup is a row of its table and the split's identity holds, since the
/// whole witness after it is the engine's run of that lookup. Its claim on `shifted` is then
/// false, and its own stage-1 instance rejects it at the stage's final check, `Reject(Sumcheck {
/// stage: 1, round: n_1 })`. A verifier that merged the two claims on `shifted` into one, as
/// 'Alternatives Considered' 5 would, accepts the same proof, its openings the cheat's committed
/// tables.
#[test]
fn a_mul3_lookup_on_another_column_is_rejected_at_stage_1() {
    let c = circuit(3);
    let setup = &c.setup;
    let graph = &setup.graph;
    let mul3 = graph
        .nodes
        .iter()
        .position(|n| n.op == Op::TableMap { table: TABLES.mul3 })
        .expect("round 1's Gf256Mul { 3 }");
    let shifted = graph.nodes[mul3].inputs[0];
    let gather = graph
        .nodes
        .iter()
        .find(|n| n.outputs == [shifted])
        .expect("ShiftRows");
    assert!(matches!(gather.op, Op::GatherPublic { .. }));
    let sub = gather.inputs[0];
    // The cheat's graph: the same tensors, its `Gf256Mul { 3 }` reading `sub`.
    let mut cheat = graph.clone();
    cheat.nodes[mul3].inputs[0] = sub;
    let cheat = Setup::new(cheat, specs());
    let witness = cheat.witness(inputs(&c, KEY, 3));
    let padded = PaddedWitness::new(&cheat.graph, &cheat.specs, &witness).expect("padded");
    let leaf = GraphLeafWitness::new(&cheat.layout, &cheat.graph, &cheat.schedule.layers, &padded)
        .expect("the cheat's leaf tensors");
    let (proof, _) = setup.prove(&leaf, &padded);
    let honest_witness = setup.witness(inputs(&c, KEY, 3));
    let public = statement(&c, &honest_witness);
    let verdict = |claims| {
        verify(
            graph,
            &setup.specs,
            &setup.schedule,
            &public,
            &proof,
            claims,
        )
    };
    assert_eq!(
        verdict(Claims::Own).err(),
        Some(Rejection::Sumcheck {
            stage: 1,
            round: setup.schedule.stages[1].rounds
        })
    );
    let merged = verdict(Claims::Merged).expect("accepted with the claims merged");
    opened(setup, &merged, &padded, &multiplicities(setup, &leaf));
    // The cheat changed what the round computes: its ciphertext is not the vector's.
    assert_ne!(
        witness
            .tensor(c.ctr.ciphertext)
            .expect("the ciphertext")
            .to_vec(),
        byte_major(&CIPHERTEXT)
    );
}

/// RFC-0072 'AES's graphs', 'Padded positions': at `B = 3` the plaintext, a committed graph
/// input, has a padded block that nothing computes; the `BitDecompose { 8, 0 }` that range-checks
/// it pins it to 0 by its input padding check. A prover that commits a padded plaintext byte of 5,
/// its bits and the ciphertext's padded byte made to agree with it, so that every lookup is a row
/// of its table and every other zero-check holds, fails stage 1's final check, `Reject(Sumcheck {
/// stage: 1, round: n_1 })`, and is accepted with exactly that check removed, its openings the
/// cheat's committed tables.
#[test]
fn a_padded_plaintext_byte_is_rejected_at_stage_1() {
    let c = circuit(3);
    let setup = &c.setup;
    let witness = setup.witness(inputs(&c, KEY, 3));
    let padded = PaddedWitness::new(&setup.graph, &setup.specs, &witness).expect("padded");
    let public = statement(&c, &witness);
    let plaintext = c.inputs[3];
    let bits = setup
        .graph
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::BitDecompose { .. }) && n.inputs == [plaintext])
        .expect("the plaintext's range check")
        .outputs[0];
    let (mut text, mut b, mut ciphertext) = (
        committed(&setup.graph, &padded, plaintext),
        committed(&setup.graph, &padded, bits),
        committed(&setup.graph, &padded, c.ctr.ciphertext),
    );
    let keystream = committed(&setup.graph, &padded, c.ctr.keystream);
    // Byte 6 of the padded block, index `4·6 + 3` of `[16, 4]`.
    let at = 4 * 6 + 3;
    text[at] = 5;
    for j in 0..8 {
        b[8 * at + j] = (5 >> j) & 1;
    }
    ciphertext[at] = 5 ^ keystream[at];
    let source = Committing {
        inner: &padded,
        tables: vec![(plaintext, text), (bits, b), (c.ctr.ciphertext, ciphertext)],
    };
    let leaf = GraphLeafWitness::new(&setup.layout, &setup.graph, &setup.schedule.layers, &source)
        .expect("the leaf tensors");
    let (proof, _) = setup.prove(&leaf, &source);
    let schedule = &setup.schedule;
    assert_eq!(
        verify(
            &setup.graph,
            &setup.specs,
            schedule,
            &public,
            &proof,
            Claims::Own
        )
        .err(),
        Some(Rejection::Sumcheck {
            stage: 1,
            round: schedule.stages[1].rounds
        })
    );
    let [check] = schedule.stages[1]
        .instances
        .iter()
        .enumerate()
        .filter(|(_, i)| i.relation == Relation::InputPadding)
        .map(|(k, _)| k)
        .collect::<Vec<_>>()[..]
    else {
        panic!("one input padding check")
    };
    let reduced = bits::without(schedule, check);
    let (proof, _) = setup.prove_with(&reduced, &leaf, &source);
    let verdict = verify(
        &setup.graph,
        &setup.specs,
        &reduced,
        &public,
        &proof,
        Claims::Own,
    )
    .expect("accepted without the input padding check");
    opened_under(
        setup,
        &reduced,
        &verdict,
        &source,
        &multiplicities(setup, &leaf),
    );
}
