// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The anchor's SHA-256 circuit (RFC-0015 'Circuits' step 3, RFC-0066, #143), lowered by
//! `tessera_compiler::anchor` and run on the CPU engine:
//! - **Known answers.** NIST CAVP's 65 short messages in one graph of two blocks, and its 64 long
//!   messages, of 3 to 101 blocks, block by block through the chaining input. A message of three
//!   blocks in one graph checks the chaining inside a graph, and its digest binds.
//! - **The key commitment.** The circuit's digest of `label ‖ k_s ‖ salt_k`, the key and salt
//!   committed, equals the recorder's `key_commitment` and `sampler_commitment`.
//! - **Binding.** A digest bit other than the computed one fails its `AssertEqual`
//!   (`ERR-WITNESS-ANCHOR`).
//! - **Uniqueness** (RFC-0066 Lemma 1). On the honest witness every `BitDecompose` passes
//!   booleanity and recomposition, over Goldilocks. A wrong carry bit, a wrong result bit, a
//!   carry made up for by a non-Boolean result bit, and two bits traded for a non-Boolean one
//!   each break one of the two identities, so a zero-check rejects them. A committed byte of
//!   value 256 has no passing bits, and the engine refuses it.
//! - **Ranges and cost.** The range analysis passes the circuit; it commits 6,376 bits per block
//!   and 8 per committed byte.

use std::path::PathBuf;

use sha2::Digest;

use tessera_anchor::{key_commitment, sampler_commitment};
use tessera_codec::error::ErrorCode;
use tessera_codec::{Secret, SecretBuf};
use tessera_compiler::anchor::{
    SHA256_BITS_PER_BLOCK, SHA256_IV, SHA256_MASKS, Sha256, Sha256Layout, sha256, sha256_bind,
    sha256_bits, sha256_blocks, sha256_digest_input, sha256_inputs, sha256_pad,
};
use tessera_compiler::{analyze, settle};
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_ir::{Graph, GraphBuilder, IrError, Node, Op, TensorId};
use tessera_witness::{Engine, Witness, WitnessError};

struct Vector {
    msg: Vec<u8>,
    md: [u8; 32],
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// The cases of a CAVP response file of `spec/vectors/sha256/`.
fn vectors(name: &str) -> Vec<Vector> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/sha256")
        .join(name);
    let text = std::fs::read_to_string(path).expect("the vector file");
    let (mut len, mut msg) = (0usize, Vec::new());
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("Len = ") {
            len = v.trim().parse::<usize>().expect("a length") / 8;
        } else if let Some(v) = line.strip_prefix("Msg = ") {
            msg = hex(v.trim());
            msg.truncate(len);
        } else if let Some(v) = line.strip_prefix("MD = ") {
            let md = hex(v.trim()).try_into().expect("32 bytes");
            out.push(Vector {
                msg: std::mem::take(&mut msg),
                md,
            });
        }
    }
    out
}

fn buffer(values: &[i64]) -> SecretBuf<i64> {
    let mut b = SecretBuf::with_capacity(values.len());
    for &v in values {
        b.push(v);
    }
    b
}

fn iv_bytes() -> Vec<u8> {
    SHA256_IV.iter().flat_map(|w| w.to_be_bytes()).collect()
}

/// `start` for `m` messages from the IV.
fn iv_bits(m: usize) -> Vec<i64> {
    let iv = iv_bytes();
    sha256_bits(&vec![iv.as_slice(); m])
}

fn masks() -> Vec<i64> {
    SHA256_MASKS.iter().flatten().copied().collect()
}

/// A circuit, its inputs, and, when `bind`, its digest input bound to the last block's words.
struct Circuit {
    graph: Graph,
    inputs: Vec<TensorId>,
    sha: Sha256,
    digest: Option<TensorId>,
}

fn circuit(layout: &Sha256Layout, bind: bool) -> Circuit {
    let mut g = GraphBuilder::new();
    let inputs: Vec<TensorId> = sha256_inputs(layout)
        .into_iter()
        .map(|decl| g.input(decl).expect("an input"))
        .collect();
    let sha = sha256(&mut g, layout, &inputs).expect("the circuit");
    let digest = bind.then(|| {
        let digest = g
            .input(sha256_digest_input(layout.messages))
            .expect("an input");
        let last = sha.chaining.last().expect("a block");
        sha256_bind(&mut g, last, digest, layout.messages).expect("the binding");
        digest
    });
    let mut graph = g.finish();
    let ranges = analyze(&graph, &[]).expect("every decomposition's input in its width");
    settle(&mut graph, &ranges).expect("bounded ranges");
    Circuit {
        graph,
        inputs,
        sha,
        digest,
    }
}

impl Circuit {
    /// The witness on padded messages, the chaining value `start`, and, when bound, `digests`.
    fn run(
        &self,
        layout: &Sha256Layout,
        padded: &[&[u8]],
        start: &[i64],
        digests: &[[u8; 32]],
    ) -> Result<Witness, WitnessError> {
        let mut values = vec![
            (self.inputs[0], buffer(&layout.public_bits(padded))),
            (self.inputs[1], buffer(start)),
            (self.inputs[2], buffer(&masks())),
        ];
        if !layout.committed.is_empty() {
            values.push((self.inputs[3], buffer(&layout.committed_bytes(padded))));
            values.push((self.inputs[4], buffer(&layout.place())));
        }
        if let Some(digest) = self.digest {
            let rows: Vec<&[u8]> = digests.iter().map(<[u8; 32]>::as_slice).collect();
            values.push((digest, buffer(&sha256_bits(&rows))));
        }
        Engine::for_graph(&self.graph, &[])
            .expect("no tables")
            .evaluate(values)
    }
}

fn values(witness: &Witness, tensor: TensorId) -> Vec<i64> {
    witness.tensor(tensor).expect("the tensor").to_vec()
}

/// Message `k`'s chaining value in `words`, each `[32, M]`, as 32 big-endian bytes.
fn digest_of(witness: &Witness, words: &[TensorId; 8], k: usize, m: usize) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (j, &word) in words.iter().enumerate() {
        let bits = values(witness, word);
        let w = (0..32).fold(0u32, |acc, i| {
            let bit = bits[i * m + k];
            assert!(bit == 0 || bit == 1, "a bit");
            acc | (u32::try_from(bit).expect("a bit") << i)
        });
        out[4 * j..4 * j + 4].copy_from_slice(&w.to_be_bytes());
    }
    out
}

/// The chaining words' bits as the next `start`, `[256, M]`.
fn chaining_bits(witness: &Witness, words: &[TensorId; 8]) -> Vec<i64> {
    words.iter().flat_map(|&w| values(witness, w)).collect()
}

/// Each message padded and extended with zero blocks to `blocks`.
fn padded_to(messages: &[&[u8]], blocks: usize) -> Vec<Vec<u8>> {
    messages
        .iter()
        .map(|m| {
            let mut p = sha256_pad(m);
            p.resize(64 * blocks, 0);
            p
        })
        .collect()
}

fn refs(rows: &[Vec<u8>]) -> Vec<&[u8]> {
    rows.iter().map(Vec::as_slice).collect()
}

fn layout(blocks: usize, messages: usize, committed: Vec<u32>) -> Sha256Layout {
    Sha256Layout {
        blocks: u32::try_from(blocks).expect("small"),
        messages: u32::try_from(messages).expect("small"),
        committed,
    }
}

#[test]
fn cavp_short_messages() {
    let cases = vectors("SHA256ShortMsg.rsp");
    assert_eq!(cases.len(), 65);
    let messages: Vec<&[u8]> = cases.iter().map(|c| c.msg.as_slice()).collect();
    let blocks: Vec<usize> = messages.iter().map(|m| sha256_blocks(m.len())).collect();
    assert!(blocks.contains(&1) && blocks.contains(&2) && blocks.iter().all(|&n| n <= 2));
    let layout = layout(2, cases.len(), Vec::new());
    let c = circuit(&layout, false);
    let padded = padded_to(&messages, 2);
    let witness = c
        .run(&layout, &refs(&padded), &iv_bits(cases.len()), &[])
        .expect("a witness");
    for (k, case) in cases.iter().enumerate() {
        let words = &c.sha.chaining[blocks[k] - 1];
        assert_eq!(
            digest_of(&witness, words, k, cases.len()),
            case.md,
            "message of {} bytes",
            case.msg.len()
        );
    }
}

#[test]
fn cavp_long_messages() {
    // One block per graph, every message at once; a message's chaining value after its last
    // block is its digest. Finished messages hash zero blocks, whose values are ignored.
    let cases = vectors("SHA256LongMsg.rsp");
    assert_eq!(cases.len(), 64);
    let m = cases.len();
    let messages: Vec<&[u8]> = cases.iter().map(|c| c.msg.as_slice()).collect();
    let blocks: Vec<usize> = messages.iter().map(|x| sha256_blocks(x.len())).collect();
    let most = *blocks.iter().max().expect("a case");
    assert_eq!((blocks.iter().min(), most), (Some(&3), 101));
    let padded = padded_to(&messages, most);
    let layout = layout(1, m, Vec::new());
    let c = circuit(&layout, false);
    let mut start = iv_bits(m);
    let mut digests = vec![None; m];
    for n in 0..most {
        let block: Vec<&[u8]> = padded.iter().map(|p| &p[64 * n..64 * (n + 1)]).collect();
        let witness = c.run(&layout, &block, &start, &[]).expect("a witness");
        let words = &c.sha.chaining[0];
        for k in (0..m).filter(|&k| blocks[k] == n + 1) {
            digests[k] = Some(digest_of(&witness, words, k, m));
        }
        start = chaining_bits(&witness, words);
    }
    for (k, case) in cases.iter().enumerate() {
        assert_eq!(
            digests[k],
            Some(case.md),
            "message of {} bytes",
            case.msg.len()
        );
    }
}

#[test]
fn a_three_block_graph_chains_and_binds_its_digest() {
    let case = &vectors("SHA256LongMsg.rsp")[0];
    assert_eq!(sha256_blocks(case.msg.len()), 3);
    let layout = layout(3, 1, Vec::new());
    let c = circuit(&layout, true);
    let padded = vec![sha256_pad(&case.msg)];
    let witness = c
        .run(&layout, &refs(&padded), &iv_bits(1), &[case.md])
        .expect("a witness that meets its digest");
    let last = c.sha.chaining.last().expect("a block");
    assert_eq!(digest_of(&witness, last, 0, 1), case.md);
}

#[test]
fn a_wrong_digest_bit_fails_its_assert_equal() {
    let cases: Vec<Vector> = vectors("SHA256ShortMsg.rsp")
        .into_iter()
        .filter(|c| sha256_blocks(c.msg.len()) == 1)
        .collect();
    assert_eq!(cases.len(), 56);
    let messages: Vec<&[u8]> = cases.iter().map(|c| c.msg.as_slice()).collect();
    let layout = layout(1, cases.len(), Vec::new());
    let c = circuit(&layout, true);
    let padded = padded_to(&messages, 1);
    let mut mds: Vec<[u8; 32]> = cases.iter().map(|c| c.md).collect();
    c.run(&layout, &refs(&padded), &iv_bits(cases.len()), &mds)
        .expect("every digest meets its message's");
    for (k, byte, bit) in [(0, 0, 0), (17, 13, 5), (55, 31, 7)] {
        mds[k][byte] ^= 1 << bit;
        let error = c
            .run(&layout, &refs(&padded), &iv_bits(cases.len()), &mds)
            .expect_err("a digest that is not the message's");
        assert!(matches!(error, WitnessError::Unequal { .. }), "{error}");
        assert_eq!(error.code(), ErrorCode::WitnessAnchor);
        mds[k][byte] ^= 1 << bit;
    }
}

/// `label ‖ key ‖ salt`, with the key's and salt's offsets.
fn commitment_message(label: &[u8], key: [u8; 16], salt: [u8; 16]) -> (Vec<u8>, Vec<u32>) {
    let mut message = label.to_vec();
    message.extend_from_slice(&key);
    message.extend_from_slice(&salt);
    let start = u32::try_from(label.len()).expect("small");
    (message, (start..start + 32).collect())
}

fn seeded_bytes(seed: &mut u64) -> [u8; 16] {
    let mut out = [0u8; 16];
    for byte in &mut out {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *byte = (*seed >> 56) as u8;
    }
    out
}

#[test]
fn key_commitments_match_the_recorders() {
    // Four sessions in one graph per label: the key and salt committed, the label public.
    let mut seed = 0x5eed;
    let keys: Vec<([u8; 16], [u8; 16])> = (0..4)
        .map(|_| (seeded_bytes(&mut seed), seeded_bytes(&mut seed)))
        .collect();
    type Commit = fn(&Secret<[u8; 16]>, &Secret<[u8; 16]>) -> [u8; 32];
    let labels: [(&[u8], Commit); 2] = [
        (b"tessera-anchor-key/v1", key_commitment),
        (b"tessera-sampler-key/v1", sampler_commitment),
    ];
    for (label, commit) in labels {
        let messages: Vec<(Vec<u8>, Vec<u32>)> = keys
            .iter()
            .map(|&(k, s)| commitment_message(label, k, s))
            .collect();
        assert!(messages.iter().all(|(m, _)| sha256_blocks(m.len()) == 1));
        let layout = layout(1, keys.len(), messages[0].1.clone());
        let c = circuit(&layout, true);
        let padded: Vec<Vec<u8>> = messages.iter().map(|(m, _)| sha256_pad(m)).collect();
        let direct: Vec<[u8; 32]> = keys
            .iter()
            .map(|&(k, s)| commit(&Secret::new(k), &Secret::new(s)))
            .collect();
        let witness = c
            .run(&layout, &refs(&padded), &iv_bits(keys.len()), &direct)
            .expect("the circuit's digests are the recorder's");
        for (k, d) in direct.iter().enumerate() {
            assert_eq!(&digest_of(&witness, &c.sha.chaining[0], k, keys.len()), d);
        }
        // The committed bits: 6,376 per block and 8 per committed byte.
        let committed: usize = c
            .graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::BitDecompose { .. }))
            .flat_map(|n| &n.outputs)
            .map(|&t| values(&witness, t).len())
            .sum();
        let per_message = SHA256_BITS_PER_BLOCK + 8 * 32;
        assert_eq!(committed as u64, per_message * keys.len() as u64);
    }
}

#[test]
fn a_frame_commits_bytes_across_its_blocks() {
    // A frame of n = 3 tokens: a public 41-byte header, then 22 + 4n committed bytes, which run
    // from block 1 into block 2 (RFC-0066 'For the anchor'), against the sha2 crate.
    let mut seed = 0xf4a3e;
    let frames: Vec<Vec<u8>> = (0..3)
        .map(|_| {
            let mut frame = Vec::new();
            while frame.len() < 41 + 34 {
                frame.extend_from_slice(&seeded_bytes(&mut seed));
            }
            frame.truncate(41 + 34);
            frame
        })
        .collect();
    assert_eq!(sha256_blocks(frames[0].len()), 2);
    let layout = layout(2, frames.len(), (41..75).collect());
    let c = circuit(&layout, true);
    let padded: Vec<Vec<u8>> = frames.iter().map(|f| sha256_pad(f)).collect();
    let direct: Vec<[u8; 32]> = frames
        .iter()
        .map(|f| sha2::Sha256::digest(f).into())
        .collect();
    let witness = c
        .run(&layout, &refs(&padded), &iv_bits(frames.len()), &direct)
        .expect("each frame's digest");
    let last = c.sha.chaining.last().expect("a block");
    for (k, d) in direct.iter().enumerate() {
        assert_eq!(&digest_of(&witness, last, k, frames.len()), d);
    }
}

// ----- Lemma 1 ----------------------------------------------------------------------------------

fn field(x: i64) -> Goldilocks {
    Goldilocks::from_i64(x)
}

/// Booleanity and recomposition of element `e` of a `BitDecompose { low, high }`, over
/// Goldilocks: `b(1 − b) = 0` for every bit, and `x = Σ 2^i b_i + 2^{low} Σ 2^j h_j`.
fn identities(low: usize, high: usize, x: &[i64], b: &[i64], h: &[i64], e: usize) -> (bool, bool) {
    let bits = b[e * low..][..low].iter().chain(&h[e * high..][..high]);
    let boolean = bits
        .clone()
        .all(|&v| field(v) * (Goldilocks::ONE - field(v)) == Goldilocks::ZERO);
    let mut sum = Goldilocks::ZERO;
    let mut weight = Goldilocks::ONE;
    for &v in bits {
        sum += weight * field(v);
        weight += weight;
    }
    (boolean, field(x[e]) - sum == Goldilocks::ZERO)
}

/// A `BitDecompose`'s low and high bit counts, input, and outputs on the witness.
fn decomposition(node: &Node, witness: &Witness) -> (usize, usize, Vec<i64>, Vec<i64>, Vec<i64>) {
    let Op::BitDecompose { low, high } = node.op else {
        panic!("a BitDecompose");
    };
    let h = node
        .outputs
        .get(1)
        .map_or_else(Vec::new, |&t| values(witness, t));
    (
        usize::from(low),
        usize::from(high),
        values(witness, node.inputs[0]),
        values(witness, node.outputs[0]),
        h,
    )
}

#[test]
fn every_decomposition_is_unique_and_a_wrong_carry_breaks_one() {
    let (message, committed) = commitment_message(b"tessera-anchor-key/v1", [7; 16], [9; 16]);
    let layout = layout(1, 2, committed);
    let c = circuit(&layout, false);
    let padded = vec![sha256_pad(&message), sha256_pad(&message)];
    let witness = c
        .run(&layout, &refs(&padded), &iv_bits(2), &[])
        .expect("a witness");
    let nodes: Vec<&Node> = c
        .graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::BitDecompose { .. }))
        .collect();
    // The committed bytes, 48 schedule words, 128 round words and 8 chaining words.
    assert_eq!(nodes.len(), 1 + 48 + 128 + 8);
    for node in &nodes {
        let (low, high, x, b, h) = decomposition(node, &witness);
        for e in 0..x.len() {
            assert_eq!(identities(low, high, &x, &b, &h, e), (true, true));
        }
    }
    // A round's decomposition of 32 bits and 3 of carry, at an element whose carry is not 7.
    let node = nodes
        .iter()
        .find(|n| n.op == Op::BitDecompose { low: 32, high: 3 })
        .expect("a round word");
    let (low, high, x, b, h) = decomposition(node, &witness);
    let e = (0..x.len())
        .find(|&e| h[e * high] == 0)
        .expect("an even carry");
    // A wrong carry bit: the result bits can no longer make up the sum.
    let mut wrong = h.clone();
    wrong[e * high] = 1;
    assert_eq!(identities(low, high, &x, &b, &wrong, e), (true, false));
    // The carry made up for in the field by a result bit of value b − 2^{32}: not a bit.
    let mut made_up = b.clone();
    made_up[e * low] -= 1 << 32;
    assert_eq!(
        identities(low, high, &x, &made_up, &wrong, e),
        (false, true)
    );
    // A wrong result bit.
    let mut flipped = b.clone();
    flipped[e * low + 5] ^= 1;
    assert_eq!(identities(low, high, &x, &flipped, &h, e), (true, false));
    // Bit i + 1 traded for 2 at bit i: the sum holds, booleanity does not.
    let i = (0..low - 1)
        .find(|&i| b[e * low + i + 1] == 1)
        .expect("a set bit");
    let mut traded = b.clone();
    traded[e * low + i + 1] = 0;
    traded[e * low + i] += 2;
    assert_eq!(identities(low, high, &x, &traded, &h, e), (false, true));
}

#[test]
fn a_committed_byte_of_256_has_no_bits() {
    // No eight bits recompose to 256 in the field: their sum is at most 255.
    let none = (0..256).all(|v: i64| {
        let b: Vec<i64> = (0..8).map(|i| (v >> i) & 1).collect();
        identities(8, 0, &[256], &b, &[], 0) == (true, false)
    });
    assert!(none);
    // And the engine refuses to decompose it.
    let (message, committed) = commitment_message(b"tessera-anchor-key/v1", [0; 16], [0; 16]);
    let layout = layout(1, 1, committed);
    let c = circuit(&layout, false);
    let padded = vec![sha256_pad(&message)];
    let mut values = vec![
        (c.inputs[0], buffer(&layout.public_bits(&refs(&padded)))),
        (c.inputs[1], buffer(&iv_bits(1))),
        (c.inputs[2], buffer(&masks())),
    ];
    let mut bytes = layout.committed_bytes(&refs(&padded));
    bytes[3] = 256;
    values.push((c.inputs[3], buffer(&bytes)));
    values.push((c.inputs[4], buffer(&layout.place())));
    let error = Engine::for_graph(&c.graph, &[])
        .expect("no tables")
        .evaluate(values)
        .expect_err("a byte of 256");
    // The circuit declares its committed bytes [0, 255]; the engine's own refusal of an integer
    // outside its decomposition's width is tested on a wider declaration in tessera-witness.
    assert_eq!(
        error,
        WitnessError::Usage {
            rule: "every input value lies in its tensor's range"
        }
    );
    assert_eq!(error.code(), ErrorCode::Usage);
}

#[test]
fn a_layout_of_unordered_or_outside_offsets_is_refused() {
    let refused = IrError::Declaration {
        rule: "the committed offsets ascend inside the padded message",
    };
    for (blocks, committed) in [
        (1, vec![5, 3]),
        (1, vec![3, 3]),
        (1, vec![64]),
        (2, vec![128]),
    ] {
        let layout = layout(blocks, 1, committed);
        let mut g = GraphBuilder::new();
        let inputs: Vec<TensorId> = sha256_inputs(&layout)
            .into_iter()
            .map(|decl| g.input(decl).expect("an input"))
            .collect();
        assert_eq!(sha256(&mut g, &layout, &inputs).err(), Some(refused));
    }
    let layout = layout(2, 1, vec![127]);
    let c = circuit(&layout, false);
    assert!(!c.graph.nodes.is_empty());
}

#[test]
fn the_range_analysis_passes_the_circuit() {
    // RFC-0066 'Committed bits per block': the least carries, c = 2, 3, 3 and 1.
    assert_eq!(SHA256_BITS_PER_BLOCK, 6_376);
    for committed in [Vec::new(), (21..53).collect()] {
        for blocks in [1, 2] {
            let layout = layout(blocks, 3, committed.clone());
            let c = circuit(&layout, true);
            analyze(&c.graph, &[]).expect("every bit decomposition's input in its width");
        }
    }
}
