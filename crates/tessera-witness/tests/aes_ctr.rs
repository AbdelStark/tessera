// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The anchor's AES-128-CTR circuit (RFC-0015 'Circuits' step 2, #142), lowered by
//! `tessera_compiler::anchor` and run on the CPU engine:
//! - **Known answers.** FIPS-197's key expansion (Appendix A.1) and its two cipher examples
//!   (Appendix B, C.1), and NIST SP 800-38A's CTR-AES128 vector (F.5.1), four blocks.
//! - **A reference.** 64 random keys with up to 9 counter blocks each, against the `aes` crate.
//! - **Ranges.** The range analysis bounds every tensor of the circuit, so every lookup's key is
//!   a true integer its table accepts (RFC-0006 'Range analysis').
//! - **Uniqueness.** Every lookup of the honest witness is a row of its table. A ciphertext with
//!   one bit flipped, or a key with one bit flipped under the ciphertext the log pins, leaves a
//!   lookup that is no row. LogUp rejects such a lookup at its roots (`tessera-logup`'s
//!   `rejection.rs`, `LookupError::Unbalanced`), so neither has a passing witness.
//! - **Cost.** The circuit's lookups are 720 for the key schedule and 1,072 per block: 268 per
//!   four-byte token.

use aes::Aes128;
use aes::cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray};
use tessera_codec::SecretBuf;
use tessera_compiler::analyze;
use tessera_compiler::anchor::{
    AesCtr, AesTables, LOOKUPS_PER_BLOCK, LOOKUPS_PER_KEY, RCON, aes128_ctr, aes128_ctr_inputs,
};
use tessera_ir::{Graph, GraphBuilder, Op, TableId, TensorId};
use tessera_tables::{BitOp, GeneratedTable, Table, TableKind, TableSpec};
use tessera_witness::{Engine, Witness};

const TABLES: AesTables = AesTables {
    sbox: TableId(0),
    mul2: TableId(1),
    mul3: TableId(2),
    xor: TableId(3),
};

fn tables() -> Vec<GeneratedTable> {
    [
        (TABLES.sbox, TableKind::AesSbox),
        (TABLES.mul2, TableKind::Gf256Mul { c: 2 }),
        (TABLES.mul3, TableKind::Gf256Mul { c: 3 }),
        (TABLES.xor, TableKind::Bitwise8 { op: BitOp::Xor }),
    ]
    .into_iter()
    .map(|(id, kind)| GeneratedTable::new(id, kind).expect("a table"))
    .collect()
}

fn specs() -> Vec<TableSpec> {
    tables().iter().map(|t| t.spec().clone()).collect()
}

/// The circuit over `blocks` counter blocks, with its inputs.
fn circuit(blocks: u32) -> (Graph, [TensorId; 4], AesCtr) {
    let mut g = GraphBuilder::new();
    let inputs = aes128_ctr_inputs(blocks).map(|decl| g.input(decl).expect("an input"));
    let ctr = aes128_ctr(&mut g, &TABLES, inputs, blocks).expect("the circuit");
    (g.finish(), inputs, ctr)
}

fn buffer(values: &[i64]) -> SecretBuf<i64> {
    let mut b = SecretBuf::with_capacity(values.len());
    for &v in values {
        b.push(v);
    }
    b
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// Blocks of 16 bytes in the byte-major layout `[16, B]`: byte `j` of block `k` at `j·B + k`.
fn byte_major(blocks: &[Vec<u8>]) -> Vec<i64> {
    let b = blocks.len();
    let mut out = vec![0i64; 16 * b];
    for (k, block) in blocks.iter().enumerate() {
        for (j, &byte) in block.iter().enumerate() {
            out[j * b + k] = i64::from(byte);
        }
    }
    out
}

/// The blocks of a `[16, B]` tensor.
fn blocks_of(values: &[i64], b: usize) -> Vec<Vec<u8>> {
    (0..b)
        .map(|k| {
            (0..16)
                .map(|j| u8::try_from(values[j * b + k]).expect("a byte"))
                .collect()
        })
        .collect()
}

fn rcon() -> Vec<i64> {
    RCON.iter().flatten().copied().collect()
}

/// The witness of the circuit on a key, counter blocks and plaintext blocks.
fn run(
    graph: &Graph,
    inputs: [TensorId; 4],
    key: &[u8],
    counters: &[Vec<u8>],
    plaintext: &[Vec<u8>],
) -> Witness {
    let engine = Engine::for_graph(graph, &specs()).expect("the tables");
    let [k, rc, c, p] = inputs;
    engine
        .evaluate(vec![
            (
                k,
                buffer(&key.iter().map(|&x| i64::from(x)).collect::<Vec<_>>()),
            ),
            (rc, buffer(&rcon())),
            (c, buffer(&byte_major(counters))),
            (p, buffer(&byte_major(plaintext))),
        ])
        .expect("a witness")
}

fn values(witness: &Witness, tensor: TensorId) -> Vec<i64> {
    witness.tensor(tensor).expect("the tensor").to_vec()
}

#[test]
fn fips_197_key_expansion_and_cipher_examples() {
    // Appendix A.1: the key 2b7e1516…, its last round key, w[40..44].
    let (graph, inputs, ctr) = circuit(1);
    let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
    let input = hex("3243f6a8885a308d313198a2e0370734");
    let witness = run(&graph, inputs, &key, &[input], &[vec![0; 16]]);
    let rk10 = values(&witness, ctr.round_keys[10]);
    assert_eq!(
        rk10,
        hex("d014f9a8c9ee2589e13f0cc8b6630ca6")
            .iter()
            .map(|&x| i64::from(x))
            .collect::<Vec<_>>()
    );
    // w[4..8], the first round key: a0fafe17 88542cb1 23a33939 2a6c7605.
    assert_eq!(
        values(&witness, ctr.round_keys[1]),
        hex("a0fafe1788542cb123a339392a6c7605")
            .iter()
            .map(|&x| i64::from(x))
            .collect::<Vec<_>>()
    );
    // Appendix B: that key on 3243f6a8…, with a zero plaintext, so the ciphertext is the
    // keystream.
    let keystream = blocks_of(&values(&witness, ctr.keystream), 1);
    assert_eq!(keystream[0], hex("3925841d02dc09fbdc118597196a0b32"));
    assert_eq!(blocks_of(&values(&witness, ctr.ciphertext), 1), keystream);
    // Appendix C.1: the key 000102…0f on 00112233…ff.
    let key = hex("000102030405060708090a0b0c0d0e0f");
    let input = hex("00112233445566778899aabbccddeeff");
    let witness = run(&graph, inputs, &key, &[input], &[vec![0; 16]]);
    assert_eq!(
        blocks_of(&values(&witness, ctr.keystream), 1)[0],
        hex("69c4e0d86a7b0430d8cdb78070b4c55a")
    );
}

#[test]
fn sp_800_38a_ctr_aes128() {
    // F.5.1 CTR-AES128.Encrypt: four blocks from the counter f0f1…feff.
    let (graph, inputs, ctr) = circuit(4);
    let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
    let counters: Vec<Vec<u8>> = [
        "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
        "f0f1f2f3f4f5f6f7f8f9fafbfcfdff00",
        "f0f1f2f3f4f5f6f7f8f9fafbfcfdff01",
        "f0f1f2f3f4f5f6f7f8f9fafbfcfdff02",
    ]
    .iter()
    .map(|s| hex(s))
    .collect();
    let plaintext: Vec<Vec<u8>> = [
        "6bc1bee22e409f96e93d7e117393172a",
        "ae2d8a571e03ac9c9eb76fac45af8e51",
        "30c81c46a35ce411e5fbc1191a0a52ef",
        "f69f2445df4f9b17ad2b417be66c3710",
    ]
    .iter()
    .map(|s| hex(s))
    .collect();
    let ciphertext: Vec<Vec<u8>> = [
        "874d6191b620e3261bef6864990db6ce",
        "9806f66b7970fdff8617187bb9fffdff",
        "5ae4df3edbd5d35e5b4f09020db03eab",
        "1e031dda2fbe03d1792170a0f3009cee",
    ]
    .iter()
    .map(|s| hex(s))
    .collect();
    let witness = run(&graph, inputs, &key, &counters, &plaintext);
    assert_eq!(blocks_of(&values(&witness, ctr.ciphertext), 4), ciphertext);
}

/// SplitMix64, the random cases' generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// `plaintext ⊕ AES_key(counter)`, block by block, by the `aes` crate.
fn reference(key: &[u8], counters: &[Vec<u8>], plaintext: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let cipher = Aes128::new(GenericArray::from_slice(key));
    counters
        .iter()
        .zip(plaintext)
        .map(|(counter, p)| {
            let mut block = GenericArray::clone_from_slice(counter);
            cipher.encrypt_block(&mut block);
            block.iter().zip(p).map(|(k, x)| k ^ x).collect()
        })
        .collect()
}

#[test]
fn random_keys_and_blocks_agree_with_the_aes_crate() {
    let mut rng = Rng(0x0142_0001);
    for b in 1..=9u32 {
        let (graph, inputs, ctr) = circuit(b);
        for _ in 0..7 {
            let key = rng.bytes(16);
            let counters: Vec<Vec<u8>> = (0..b).map(|_| rng.bytes(16)).collect();
            let plaintext: Vec<Vec<u8>> = (0..b).map(|_| rng.bytes(16)).collect();
            let witness = run(&graph, inputs, &key, &counters, &plaintext);
            assert_eq!(
                blocks_of(&values(&witness, ctr.ciphertext), b as usize),
                reference(&key, &counters, &plaintext),
                "{b} blocks"
            );
        }
    }
}

#[test]
fn the_range_analysis_bounds_every_tensor_of_the_circuit() {
    for b in [1, 3, 16] {
        let (graph, _, _) = circuit(b);
        analyze(&graph, &specs()).expect("every lookup key in its table's domain");
    }
}

/// Every lookup of `graph` in `witness`, with its table: `(table, key, value)`, the key of a
/// `Bitwise` lookup being `a + 2^8 · b` (RFC-0008).
fn lookups(graph: &Graph, witness: &Witness) -> Vec<(TableId, i64, i64)> {
    let mut out = Vec::new();
    for node in &graph.nodes {
        let (table, keys): (TableId, Vec<i64>) = match &node.op {
            Op::TableMap { table } => (*table, values(witness, node.inputs[0])),
            Op::Bitwise { table } => {
                let (a, b) = (
                    values(witness, node.inputs[0]),
                    values(witness, node.inputs[1]),
                );
                (*table, a.iter().zip(&b).map(|(a, b)| a + 256 * b).collect())
            }
            _ => continue,
        };
        let outputs = values(witness, node.outputs[0]);
        out.extend(keys.into_iter().zip(outputs).map(|(k, v)| (table, k, v)));
    }
    out
}

/// The lookups among `tuples` that are no row of their table.
fn off_table(tuples: &[(TableId, i64, i64)]) -> usize {
    let tables = tables();
    tuples
        .iter()
        .filter(|&&(table, key, value)| {
            let t = &tables[table.0 as usize];
            t.lookup(key, 0) != Some(value)
        })
        .count()
}

#[test]
fn every_honest_lookup_is_a_row_and_one_flipped_bit_leaves_one_that_is_not() {
    let (graph, inputs, ctr) = circuit(2);
    let mut rng = Rng(0x0142_0002);
    let key = rng.bytes(16);
    let counters: Vec<Vec<u8>> = (0..2).map(|_| rng.bytes(16)).collect();
    let plaintext: Vec<Vec<u8>> = (0..2).map(|_| rng.bytes(16)).collect();
    let honest = run(&graph, inputs, &key, &counters, &plaintext);
    let tuples = lookups(&graph, &honest);
    assert_eq!(off_table(&tuples), 0);
    let ciphertext = values(&honest, ctr.ciphertext);

    // The ciphertext with one bit flipped, everything else honest: its XOR is no row. The
    // ciphertext's XOR is the last node, so its lookups are the last ones.
    let last = graph.nodes.last().expect("a node");
    assert_eq!(last.outputs[0], ctr.ciphertext);
    let start = tuples.len() - ciphertext.len();
    for (byte, bit) in [(0usize, 0u32), (17, 7), (31, 3)] {
        let mut flipped = tuples.clone();
        flipped[start + byte].2 ^= 1 << bit;
        assert_eq!(off_table(&flipped), 1, "byte {byte}, bit {bit}");
    }

    // A key with one bit flipped, under the ciphertext the log pins: the witness of that key,
    // whose ciphertext is replaced by the logged one. Every byte of the keystream changes with
    // probability about 255/256, so most of the ciphertext's XORs are no row.
    for bit in [0u32, 37, 127] {
        let mut wrong = key.clone();
        wrong[(bit / 8) as usize] ^= 1 << (bit % 8);
        let other = run(&graph, inputs, &wrong, &counters, &plaintext);
        let mut forged = lookups(&graph, &other);
        let start = forged.len() - ciphertext.len();
        for (tuple, &c) in forged[start..].iter_mut().zip(&ciphertext) {
            tuple.2 = c;
        }
        let bad = off_table(&forged);
        assert!(
            bad >= 24,
            "key bit {bit}: {bad} of 32 ciphertext lookups are no row"
        );
    }
}

#[test]
fn the_circuit_costs_268_lookups_per_token() {
    for b in [1u32, 4, 16] {
        let (graph, _, _) = circuit(b);
        let count: u64 = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::TableMap { .. } | Op::Bitwise { .. }))
            .map(|n| {
                graph.tensors[n.outputs[0].0 as usize]
                    .shape
                    .iter()
                    .map(|&d| u64::from(d))
                    .product::<u64>()
            })
            .sum();
        assert_eq!(count, LOOKUPS_PER_KEY + u64::from(b) * LOOKUPS_PER_BLOCK);
    }
    assert_eq!(LOOKUPS_PER_KEY, 720);
    assert_eq!(LOOKUPS_PER_BLOCK, 1_072);
    // Four bytes per token, sixteen per block.
    assert_eq!(LOOKUPS_PER_BLOCK / 4, 268);
}
