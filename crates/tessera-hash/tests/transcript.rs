// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Fiat–Shamir transcript (RFC-0004 'Transcript', 'Testing Strategy'): the vectors of
//! `spec/vectors/transcript/`, written by an independent oracle; every rejection branch of the
//! challenge samplers, driven by a hash stub; the uniformity of `challenge_fp`; and grinding.

mod common;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;

use serde_json::Value;
use tessera_codec::{Ascii, Encode, Sink, encode};
use tessera_field::{Goldilocks, PrimeField, QuadraticExtension};
use tessera_hash::{
    Digest, Domain, Hasher, Mode, ParamSetId, SessionContext, Transcript, TranscriptHash, hash,
};

const P: u64 = Goldilocks::ORDER;

fn context(mode: Mode, params: &str) -> SessionContext {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode,
        params: ParamSetId(Ascii::new(params).expect("ASCII")),
        statement_digest: [0x5a; 32],
        session_id: [7; 16],
    }
}

/// TCE bytes absorbed as they are, as the vectors give them.
struct Raw(Vec<u8>);

impl Encode for Raw {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        out.put(&self.0);
    }
}

fn number(value: &Value) -> u64 {
    value
        .as_str()
        .expect("a decimal string")
        .parse()
        .expect("a u64")
}

fn numbers(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .expect("a list")
        .iter()
        .map(number)
        .collect()
}

#[test]
fn transcripts_match_the_independent_oracle_step_by_step() {
    let path = common::vectors().join("transcript/transcript.json");
    let file: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("vectors")).expect("JSON");
    let cases = file["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 6);
    let mut steps = 0;
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let c = &case["context"];
        let text = |key: &str| c[key].as_str().expect("a string");
        let ctx = SessionContext {
            protocol: Ascii::new(text("protocol")).expect("ASCII"),
            mode: match text("mode") {
                "Live" => Mode::Live,
                "NonInteractive" => Mode::NonInteractive,
                other => panic!("mode {other}"),
            },
            params: ParamSetId(Ascii::new(text("params")).expect("ASCII")),
            statement_digest: common::digest(text("statement_digest")),
            session_id: common::bytes(text("session_id"))
                .try_into()
                .expect("16 bytes"),
        };
        assert_eq!(
            encode(&ctx),
            common::bytes(case["context_tce"].as_str().expect("hex")),
            "{name}: TCE(SessionContext)"
        );
        let mut t = Transcript::new(&ctx);
        assert_eq!(
            t.state(),
            common::digest(case["state"].as_str().expect("hex"))
        );

        for step in case["steps"].as_array().expect("steps") {
            let label = step["label"].as_str().expect("label");
            let at = format!("{name}, {} {label:?}", step["op"]);
            match step["op"].as_str().expect("op") {
                "absorb" => t.absorb(
                    label,
                    &Raw(common::bytes(step["tce"].as_str().expect("hex"))),
                ),
                "squeeze" => assert_eq!(
                    t.squeeze(label),
                    common::digest(step["out"].as_str().expect("hex")),
                    "{at}"
                ),
                "challenge_fp" => assert_eq!(
                    t.challenge_fp(label).to_canonical_u64(),
                    number(&step["value"]),
                    "{at}"
                ),
                "challenge_fp2" => {
                    let x = t.challenge_fp2(label);
                    let [c0, c1] = x.coeffs().map(|c| c.to_canonical_u64());
                    assert_eq!(vec![c0, c1], numbers(&step["value"]), "{at}");
                }
                "challenge_indices" => {
                    let log_n = step["log_n"].as_u64().expect("log_n") as u32;
                    let count = step["count"].as_u64().expect("count") as u32;
                    assert_eq!(
                        t.challenge_indices(label, log_n, count),
                        numbers(&step["value"]),
                        "{at}"
                    );
                }
                "grind" => {
                    let bits = step["bits"].as_u64().expect("bits") as u8;
                    assert_eq!(t.grind(label, bits), number(&step["nonce"]), "{at}");
                }
                "check_grind" => {
                    let bits = step["bits"].as_u64().expect("bits") as u8;
                    let valid = step["valid"].as_bool().expect("valid");
                    assert_eq!(
                        t.check_grind(label, bits, number(&step["nonce"])),
                        valid,
                        "{at}"
                    );
                }
                other => panic!("unknown op {other}"),
            }
            assert_eq!(
                t.state(),
                common::digest(step["state"].as_str().expect("hex")),
                "{at}: state"
            );
            assert_eq!(
                t.counter(),
                step["counter"].as_u64().expect("counter"),
                "{at}: counter"
            );
            steps += 1;
        }
    }
    assert_eq!(steps, 33, "every step of the vectors");
}

/// BLAKE3, except that squeeze and proof-of-work outputs come from a script while it lasts.
#[derive(Default)]
struct Scripted {
    squeezes: RefCell<VecDeque<Digest>>,
    pows: RefCell<VecDeque<Digest>>,
}

impl Scripted {
    fn squeezing(outputs: &[[u64; 4]]) -> Self {
        let digests = outputs.iter().map(|words| {
            let mut out = [0; 32];
            for (chunk, word) in out.chunks_exact_mut(8).zip(words) {
                chunk.copy_from_slice(&word.to_le_bytes());
            }
            out
        });
        Self {
            squeezes: RefCell::new(digests.collect()),
            ..Self::default()
        }
    }
}

struct Input {
    domain: Domain,
    hasher: Hasher,
}

impl Sink for Input {
    fn put(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }
}

impl TranscriptHash for Scripted {
    type Input = Input;

    fn input(&self, domain: Domain) -> Input {
        Input {
            domain,
            hasher: Hasher::new(domain),
        }
    }

    fn digest(&self, input: Input) -> Digest {
        let script = match input.domain {
            Domain::TranscriptSqueeze => self.squeezes.borrow_mut().pop_front(),
            Domain::Pow => self.pows.borrow_mut().pop_front(),
            _ => None,
        };
        script.unwrap_or_else(|| input.hasher.finalize())
    }
}

/// A transcript over `hash`, with the state it starts from.
fn scripted(hash: Scripted) -> (Transcript<Scripted>, Digest) {
    let t = Transcript::with_hash(&context(Mode::NonInteractive, "test-small/1"), hash);
    let state = t.state();
    (t, state)
}

/// The state after squeezing `out` from `state`: the ratchet, recomputed here.
fn ratchet(state: Digest, out: [u64; 4]) -> Digest {
    let mut bytes = state.to_vec();
    for word in out {
        bytes.extend(word.to_le_bytes());
    }
    hash(Domain::TranscriptRatchet, &bytes)
}

#[test]
fn challenge_fp_scans_each_squeeze_and_squeezes_again_when_no_word_is_below_p() {
    // Word 0 is P itself, the smallest rejected value; words 1 and 2 are above it; word 3, P − 1,
    // the largest accepted value, is the challenge.
    let out = [P, P + 1, u64::MAX, P - 1];
    let (mut t, start) = scripted(Scripted::squeezing(&[out]));
    assert_eq!(t.challenge_fp("c").to_canonical_u64(), P - 1);
    assert_eq!((t.counter(), t.state()), (1, ratchet(start, out)));

    // No word below P: a second squeeze, whose first word is taken.
    let rejected = [P, P + 7, u64::MAX, P];
    let next = [5, 0, 0, 0];
    let (mut t, start) = scripted(Scripted::squeezing(&[rejected, next]));
    assert_eq!(t.challenge_fp("c").to_canonical_u64(), 5);
    assert_eq!(
        (t.counter(), t.state()),
        (2, ratchet(ratchet(start, rejected), next))
    );
}

#[test]
fn challenge_fp2_squeezes_again_unless_both_coefficients_are_below_p() {
    // c0 rejected; then c1 rejected; then both accepted, whatever words 2 and 3 hold.
    let outs = [[P, 1, 2, 3], [4, P, 5, 6], [P - 1, 0, u64::MAX, P]];
    let (mut t, start) = scripted(Scripted::squeezing(&outs));
    let x = t.challenge_fp2("c");
    assert_eq!(x.coeffs().map(|c| c.to_canonical_u64()), [P - 1, 0]);
    let state = outs.iter().fold(start, |state, &out| ratchet(state, out));
    assert_eq!((t.counter(), t.state()), (3, state));
}

#[test]
fn challenge_indices_mask_words_four_per_squeeze() {
    let a = [
        0xFFFF_FFFF_FFFF_FFFF,
        0x1234_5678_9ABC_DEF0,
        3 << 20,
        (1 << 20) - 1,
    ];
    let b = [1 << 21, 0xF0F0_F0F0_F0F0_F0F0, 7, 8];
    let c = [P, 0, 0, 0];
    // Nine indices need three squeezes; only the first word of the third is used.
    let (mut t, start) = scripted(Scripted::squeezing(&[a, b, c]));
    let mask = (1 << 20) - 1;
    let expected: Vec<u64> = a
        .iter()
        .chain(&b)
        .chain(&c[..1])
        .map(|w| w & mask)
        .collect();
    assert_eq!(t.challenge_indices("q", 20, 9), expected);
    let state = [a, b, c]
        .iter()
        .fold(start, |state, &out| ratchet(state, out));
    assert_eq!((t.counter(), t.state()), (3, state));

    // A domain of one index, and the whole of u64: the masks of 0 and 64 bits.
    let (mut t, _) = scripted(Scripted::squeezing(&[a, a]));
    assert_eq!(t.challenge_indices("q", 0, 3), vec![0, 0, 0]);
    assert_eq!(t.challenge_indices("q", 64, 4), a.to_vec());
    // No index, no squeeze.
    let before = (t.counter(), t.state());
    assert!(t.challenge_indices("q", 10, 0).is_empty());
    assert_eq!((t.counter(), t.state()), before);
}

#[test]
fn labels_are_printable_ascii() {
    // Labels are hashed as TCE `Ascii`, bytes 0x20 to 0x7E: both ends are labels, and a byte
    // outside is refused rather than hashed in a form no `Ascii` value has.
    let ctx = context(Mode::Live, "live-64/1");
    Transcript::new(&ctx).absorb(" ~", &0u64);
    for label in ["round\n1", "\u{1f}", "\u{7f}", "é"] {
        let refused = std::panic::catch_unwind(|| Transcript::new(&ctx).absorb(label, &0u64));
        assert!(refused.is_err(), "{label:?} accepted");
    }
}

#[test]
#[should_panic(expected = "exceeds u64")]
fn an_index_domain_past_u64_is_refused() {
    let mut t = Transcript::new(&context(Mode::Live, "live-64/1"));
    let _ = t.challenge_indices("q", 65, 1);
}

#[test]
fn proof_of_work_counts_leading_zero_bits_from_the_top_of_byte_0() {
    // 0x00 0x7F …: nine zero bits, read from the most significant bit of byte 0.
    let mut nine = [0xFF; 32];
    nine[0] = 0x00;
    nine[1] = 0x7F;
    // 0x80 …: no leading zero, though bit 0 of byte 0 and a whole trailing byte are zero.
    let mut none = [0x00; 32];
    none[0] = 0x80;
    for (out, bits, valid) in [
        (nine, 9, true),
        (nine, 10, false),
        (none, 0, true),
        (none, 1, false),
    ] {
        let hash = Scripted {
            pows: RefCell::new(VecDeque::from([out])),
            ..Scripted::default()
        };
        let (mut t, start) = scripted(hash);
        assert_eq!(t.check_grind("pow", bits, 42), valid, "{bits} bits");
        // A valid nonce is absorbed; an invalid one leaves the state as it was.
        let mut expected = Transcript::with_hash(
            &context(Mode::NonInteractive, "test-small/1"),
            Scripted::default(),
        );
        if valid {
            expected.absorb("pow", &42u64);
        }
        assert_eq!(t.state(), expected.state(), "{bits} bits");
        assert_ne!(valid, t.state() == start);
    }
}

#[test]
fn grinding_finds_the_smallest_nonce_and_checking_accepts_any_valid_one() {
    let ctx = context(Mode::NonInteractive, "ni-100/1");
    let fresh = || Transcript::new(&ctx);
    let mut prover = fresh();
    let nonce = prover.grind("seed-pow", 16);

    // The verifier accepts it, and both transcripts absorbed it.
    let mut verifier = fresh();
    assert!(verifier.check_grind("seed-pow", 16, nonce));
    assert_eq!(verifier.state(), prover.state());
    let mut absorbed = fresh();
    absorbed.absorb("seed-pow", &nonce);
    assert_eq!(absorbed.state(), prover.state());

    // The same nonce fails 24 bits, leaving the transcript as it was.
    let mut stricter = fresh();
    assert!(!stricter.check_grind("seed-pow", 24, nonce));
    assert_eq!(stricter.state(), fresh().state());

    // Every smaller nonce fails 16 bits, so the prover's is the smallest.
    assert!((0..nonce).all(|n| !fresh().check_grind("seed-pow", 16, n)));

    // A later valid nonce is accepted too, and absorbed in place of the first.
    let later = (nonce + 1..)
        .find(|&n| fresh().check_grind("seed-pow", 16, n))
        .expect("another valid nonce");
    let mut verifier = fresh();
    assert!(verifier.check_grind("seed-pow", 16, later));
    let mut absorbed = fresh();
    absorbed.absorb("seed-pow", &later);
    assert_eq!(verifier.state(), absorbed.state());
    assert_ne!(verifier.state(), prover.state());

    // No work is required at 0 bits: the first nonce, 0.
    assert_eq!(fresh().grind("seed-pow", 0), 0);
}

#[test]
fn challenge_fp_is_uniform_modulo_2_16() {
    // 2^20 challenges into 2^16 cells, 16 expected in each: the chi-square statistic, with 65535
    // degrees of freedom, must stay below its 0.999 quantile, 66659 by the Wilson–Hilferty
    // approximation (`k·(1 − 2/9k + z·√(2/9k))³`, z = 3.0902). The transcript is deterministic, so
    // the outcome is too.
    let mut t = Transcript::new(&context(Mode::NonInteractive, "ni-100/1"));
    let mut cells = vec![0u32; 1 << 16];
    for _ in 0..1 << 20 {
        cells[(t.challenge_fp("u").to_canonical_u64() & 0xFFFF) as usize] += 1;
    }
    let expected = f64::from(1u32 << 4);
    let chi2: f64 = cells
        .iter()
        .map(|&n| (f64::from(n) - expected).powi(2) / expected)
        .sum();
    assert!(chi2 < 66_659.0, "chi-square {chi2}");
}

#[test]
fn any_single_byte_difference_changes_every_later_challenge() {
    // 07 §3: identical absorb sequences give identical challenges, and a difference in one byte
    // of a message, a label or the context changes every challenge after it, and none before.
    let messages: Vec<(String, Vec<u8>)> = vec![
        ("round1".into(), (0..40).collect()),
        ("tree/0/round/0".into(), (40..72).collect()),
        ("stage/0/zk-sum".into(), vec![9; 16]),
        (
            "pcs/advice/combined".into(),
            (0..200u32).map(|i| (i * 7) as u8).collect(),
        ),
    ];
    let run = |ctx: &SessionContext, messages: &[(String, Vec<u8>)]| -> Vec<u64> {
        let mut t = Transcript::new(ctx);
        messages
            .iter()
            .map(|(label, bytes)| {
                t.absorb(label, &Raw(bytes.clone()));
                t.challenge_fp(label).to_canonical_u64()
            })
            .collect()
    };
    let ctx = context(Mode::NonInteractive, "ni-100/1");
    let base = run(&ctx, &messages);
    assert_eq!(run(&ctx, &messages), base);

    let changes_from = |i: usize, challenges: &[u64], what: &str| {
        assert_eq!(
            challenges[..i],
            base[..i],
            "{what}: an earlier challenge changed"
        );
        assert!(
            challenges[i..].iter().zip(&base[i..]).all(|(a, b)| a != b),
            "{what}: a later challenge did not change"
        );
    };
    for (i, (label, bytes)) in messages.iter().enumerate() {
        for at in 0..bytes.len() {
            let mut changed = messages.clone();
            changed[i].1[at] ^= 1 << (at % 8);
            changes_from(i, &run(&ctx, &changed), &format!("message {i}, byte {at}"));
        }
        let mut changed = messages.clone();
        changed[i].0 = format!("{}!", &label[..label.len() - 1]);
        changes_from(i, &run(&ctx, &changed), &format!("label {i}"));
    }
    let mut other = ctx.clone();
    other.session_id[15] ^= 1;
    changes_from(0, &run(&other, &messages), "session id");
}

#[test]
fn a_long_message_hashes_as_one_input_however_it_is_written() {
    // `absorb` collects a long message's small writes into large updates. Whatever the sizes of
    // the writes, around the first kilobyte and the 64 KiB updates, the hash must see the bytes
    // of `state ‖ Ascii(label) ‖ TCE(value)` once each and in order.
    let ctx = context(Mode::NonInteractive, "ni-100/1");
    let expected = |state: Digest, label: &str, tce: &[u8]| {
        let input = [
            &state[..],
            &(label.len() as u64).to_le_bytes(),
            label.as_bytes(),
            tce,
        ]
        .concat();
        hash(Domain::TranscriptAbsorb, &input)
    };
    /// A value encoded by the writes it is given, so that their sizes can be chosen.
    struct Writes(Vec<Vec<u8>>);
    impl Encode for Writes {
        fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
            for write in &self.0 {
                out.put(write);
            }
        }
    }
    let mut rng = common::SplitMix64::new(0x7a45_0045);
    let mut bytes = |n: usize| -> Vec<u8> { (0..n).map(|_| rng.next_u64() as u8).collect() };
    let cases: Vec<Writes> = vec![
        // Eight-byte writes, one field element at a time, up to and past each threshold.
        Writes((0..128).map(|_| bytes(8)).collect()),
        Writes((0..129).map(|_| bytes(8)).collect()),
        Writes((0..20_000).map(|_| bytes(8)).collect()),
        // Small writes, then one write as long as an update, which must not overtake them.
        Writes([(0..300).map(|_| bytes(8)).collect(), vec![bytes(70_000)]].concat()),
        // Writes of odd sizes around 64 KiB.
        Writes(vec![
            bytes(1_000),
            bytes(65_535),
            bytes(1),
            bytes(65_536),
            bytes(3),
        ]),
        Writes(vec![bytes(200_000)]),
    ];
    for (i, writes) in cases.iter().enumerate() {
        let mut t = Transcript::new(&ctx);
        let before = t.state();
        t.absorb("pcs/advice/combined", writes);
        let tce = writes.0.concat();
        assert_eq!(
            t.state(),
            expected(before, "pcs/advice/combined", &tce),
            "case {i}, {} bytes",
            tce.len()
        );
    }
    // The ordinary encoding of a long vector of field elements, one element per write.
    let values: Vec<Goldilocks> = (0..40_000u64)
        .map(|i| Goldilocks::from_canonical_u64(i * 11).expect("below P"))
        .collect();
    let mut t = Transcript::new(&ctx);
    let before = t.state();
    t.absorb("pcs/advice/combined", &values);
    assert_eq!(
        t.state(),
        expected(before, "pcs/advice/combined", &encode(&values))
    );
}
