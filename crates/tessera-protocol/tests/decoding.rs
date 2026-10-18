// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The decoders beyond the golden vectors (RFC-0003 'Limits', INV-TCE-3, RFC-0036 'Matrix ids'):
//!
//! - a hostile length allocates nothing, in the codec's sequences and in the matrix-keyed ones,
//!   as a counting allocator sees it, and a keyed sequence that fails at its first entry holds
//!   no more memory than its input;
//! - a keyed entry's matrix id is checked before anything after it is read, and every id but 0
//!   to 9 is refused in each keyed sequence;
//! - every `MIN_SIZE`, the bound the length checks multiply, is the length of the type's
//!   smallest value, the keyed entries' included, which their messages pin from both sides.
//!
//! `allocation-counter` counts the current thread's allocations only, so tests running in
//! parallel do not disturb each other.

use allocation_counter::{AllocationInfo, measure};
use tessera_codec::error::ErrorCode;
use tessera_codec::{Decode, DecodeError, DecodeLimits, Encode, decode, encode};
use tessera_protocol::{
    AdviceCommitment, BatchDescriptor, ColumnQueries, CombinedRow, Decoding, FixedBits, LogUpProof,
    MatrixColumns, MatrixQueries, MultiProof, NiProof, PredictorSpec, ProverSeedShare,
    PublicStatement, Round1, Round2Response, Round3Response, ScoringClaims, SeedCommit, SeedReveal,
    StageId, StageProof, SumcheckProof, TreeProof, WeightCombinedRows,
};

/// The outcome of decoding `input` as `T`, and what decoding allocated.
fn decoding<T: Decode>(
    input: &[u8],
    limits: DecodeLimits,
) -> (Result<(), DecodeError>, AllocationInfo) {
    let mut outcome = None;
    let info = measure(|| outcome = Some(decode::<T>(input, limits).map(drop)));
    (outcome.expect("measured"), info)
}

fn failure(code: ErrorCode, offset: u64) -> DecodeError {
    DecodeError { code, offset }
}

fn le(x: u64) -> [u8; 8] {
    x.to_le_bytes()
}

/// The four empty lists a `Round2Response` starts with when its leaf stage, stages and advice
/// rows are empty: 48 bytes of zero lengths.
fn empty_leading_lists() -> Vec<u8> {
    vec![0; 48]
}

#[test]
fn a_hostile_length_allocates_nothing() {
    // 2^63 advice columns declared, 16 bytes left: past max_vec_len.
    let input = [le(1 << 63).as_slice(), &[0; 16]].concat();
    let (outcome, info) = decoding::<Round3Response>(&input, DecodeLimits::MESSAGE);
    assert_eq!(outcome, Err(failure(ErrorCode::CodecLimit, 0)));
    assert_eq!(info.count_total, 0, "{info:?}");

    // Within an unlimited max_vec_len, a keyed sequence that the bytes left cannot hold:
    // ColumnQueries' weights, after an empty advice list, declaring 2^40 entries.
    let unlimited = DecodeLimits {
        max_vec_len: u64::MAX,
        ..DecodeLimits::MESSAGE
    };
    let input = [le(0).as_slice(), &le(1 << 40), &[0; 16]].concat();
    let (outcome, info) = decoding::<ColumnQueries>(&input, unlimited);
    assert_eq!(outcome, Err(failure(ErrorCode::CodecTruncated, 8)));
    assert_eq!(info.count_total, 0, "{info:?}");

    // The same in Round2Response's weight_combined and Round3Response's weight_columns.
    let input = [empty_leading_lists().as_slice(), &le(u64::MAX), &[0; 40]].concat();
    let (outcome, info) = decoding::<Round2Response>(&input, unlimited);
    assert_eq!(outcome, Err(failure(ErrorCode::CodecTruncated, 48)));
    assert_eq!(info.count_total, 0, "{info:?}");
    let input = [[0; 16].as_slice(), &le(1 << 62), &[0; 32]].concat();
    let (outcome, info) = decoding::<Round3Response>(&input, unlimited);
    assert_eq!(outcome, Err(failure(ErrorCode::CodecTruncated, 16)));
    assert_eq!(info.count_total, 0, "{info:?}");
}

/// A `WeightCombinedRows` takes about three times its smallest encoding in memory: a
/// reservation sized by the declared count would take three times the input. The first entry's
/// id, 0xFF, is refused, after the reservation.
#[test]
fn a_keyed_sequence_failing_at_its_first_entry_holds_no_more_than_its_input() {
    let entries = 1 << 12;
    let mut input = empty_leading_lists();
    input.extend(le(entries));
    // An entry's smallest encoding: its id and four empty lists, 1 + 4 · 8 bytes.
    input.extend(vec![0xFF; entries as usize * 33]);
    let (outcome, info) = decoding::<Round2Response>(&input, DecodeLimits::MESSAGE);
    assert_eq!(outcome, Err(failure(ErrorCode::CodecNoncanonical, 56)));
    assert!(info.bytes_max > 0, "nothing was reserved: {info:?}");
    assert!(info.bytes_max <= input.len() as u64, "{info:?}");
}

/// An entry's id is its first byte, read and checked before the rest of the entry, as a map's
/// key is: an id that names no matrix, or repeats the previous one, is refused even when a
/// hostile length follows it. A valid id lets that length be read and refused.
#[test]
fn an_entrys_matrix_id_is_checked_before_the_rest_of_it() {
    let limits = DecodeLimits::MESSAGE;
    // ColumnQueries { advice: [], weights: [entry] }, the entry's indices declaring 2^64 − 1.
    let one = |id: u8| [le(0).as_slice(), &le(1), &[id], &le(u64::MAX)].concat();
    assert_eq!(
        decode::<ColumnQueries>(&one(12), limits).map(drop),
        Err(failure(ErrorCode::CodecNoncanonical, 16))
    );
    assert_eq!(
        decode::<ColumnQueries>(&one(2), limits).map(drop),
        Err(failure(ErrorCode::CodecLimit, 17))
    );
    // Two entries, the second repeating the first's id.
    let two = |second: u8| {
        [
            le(0).as_slice(),
            &le(2),
            &[1],
            &le(0),
            &[second],
            &le(u64::MAX),
        ]
        .concat()
    };
    assert_eq!(
        decode::<ColumnQueries>(&two(1), limits).map(drop),
        Err(failure(ErrorCode::CodecNoncanonical, 25))
    );
    assert_eq!(
        decode::<ColumnQueries>(&two(3), limits).map(drop),
        Err(failure(ErrorCode::CodecLimit, 26))
    );
}

/// One entry with `id` in each keyed sequence: ids 0 to 9 decode, and every other is
/// `ERR-CODEC-NONCANONICAL` at the id's byte, the entry's first.
#[test]
fn every_id_but_0_to_9_is_refused_in_each_keyed_sequence() {
    let limits = DecodeLimits::MESSAGE;
    let row = CombinedRow {
        u: vec![],
        mask: vec![],
    };
    for id in 0..=u8::MAX {
        let queries = ColumnQueries {
            advice: vec![],
            weights: vec![MatrixQueries {
                matrix: id,
                indices: vec![3],
            }],
        };
        let mut round2 = empty_round2();
        round2.weight_combined = vec![WeightCombinedRows {
            matrix: id,
            evaluation: row.clone(),
            proximity: row.clone(),
        }];
        let round3 = Round3Response {
            advice_columns: vec![],
            advice_proof: MultiProof::default(),
            weight_columns: vec![MatrixColumns {
                matrix: id,
                columns: vec![],
                proof: MultiProof::default(),
            }],
        };
        // Where each entry starts: after the lists before it and the sequence's length.
        let outcomes = [
            (
                decode::<ColumnQueries>(&encode(&queries), limits).map(drop),
                16,
            ),
            (
                decode::<Round2Response>(&encode(&round2), limits).map(drop),
                56,
            ),
            (
                decode::<Round3Response>(&encode(&round3), limits).map(drop),
                24,
            ),
        ];
        for (outcome, at) in outcomes {
            let expected = if id < 10 {
                Ok(())
            } else {
                Err(failure(ErrorCode::CodecNoncanonical, at))
            };
            assert_eq!(outcome, expected, "id {id}");
        }
    }
}

/// A `Round2Response` with every list empty.
fn empty_round2() -> Round2Response {
    Round2Response {
        logup: LogUpProof {
            roots: vec![],
            values: vec![],
        },
        trees: TreeProof {
            levels: vec![],
            splits: vec![],
        },
        stages: vec![],
        advice_combined: vec![],
        weight_combined: vec![],
        scoring: ScoringClaims { requests: vec![] },
    }
}

/// Nine entries of each keyed type at their smallest, ids 0 to 8, fill exactly the bytes their
/// sequence's length check reserves: they decode, and one byte fewer is `ERR-CODEC-TRUNCATED` at
/// the sequence's length. An entry's `MIN_SIZE` one byte too large fails the first; one byte
/// too small lets the second through the length check, to fail inside an entry.
#[test]
fn keyed_entries_at_their_smallest_fill_their_sequence_exactly() {
    let limits = DecodeLimits::MESSAGE;
    let row = CombinedRow {
        u: vec![],
        mask: vec![],
    };
    let queries = ColumnQueries {
        advice: vec![],
        weights: (0..9)
            .map(|matrix| MatrixQueries {
                matrix,
                indices: vec![],
            })
            .collect(),
    };
    let mut round2 = empty_round2();
    round2.weight_combined = (0..9)
        .map(|matrix| WeightCombinedRows {
            matrix,
            evaluation: row.clone(),
            proximity: row.clone(),
        })
        .collect();
    let round3 = Round3Response {
        advice_columns: vec![],
        advice_proof: MultiProof::default(),
        weight_columns: (0..9)
            .map(|matrix| MatrixColumns {
                matrix,
                columns: vec![],
                proof: MultiProof::default(),
            })
            .collect(),
    };
    // Each message's bytes, the sequence's offset, its entries' smallest size, and what follows
    // the entries: Round2Response's empty scoring claims, which the short input leaves out.
    let cases: [(&str, Vec<u8>, u64, usize, usize); 3] = [
        ("ColumnQueries", encode(&queries), 8, 9, 0),
        ("Round2Response", encode(&round2), 48, 33, 8),
        ("Round3Response", encode(&round3), 16, 17, 0),
    ];
    for (name, bytes, at, entry, after) in cases {
        assert_eq!(
            bytes.len() as u64,
            at + 8 + 9 * entry as u64 + after as u64,
            "{name}"
        );
        let short = &bytes[..bytes.len() - after - 1];
        let outcome = match name {
            "ColumnQueries" => (
                decode::<ColumnQueries>(&bytes, limits).map(drop),
                decode::<ColumnQueries>(short, limits).map(drop),
            ),
            "Round2Response" => (
                decode::<Round2Response>(&bytes, limits).map(drop),
                decode::<Round2Response>(short, limits).map(drop),
            ),
            _ => (
                decode::<Round3Response>(&bytes, limits).map(drop),
                decode::<Round3Response>(short, limits).map(drop),
            ),
        };
        assert_eq!(outcome.0, Ok(()), "{name}");
        assert_eq!(
            outcome.1,
            Err(failure(ErrorCode::CodecTruncated, at)),
            "{name}"
        );
    }
}

/// The smallest value of each type encodes to exactly its `MIN_SIZE`, so the length checks
/// neither refuse valid input nor let a length through that the input cannot hold.
#[test]
fn min_size_is_the_length_of_the_smallest_value() {
    fn smallest<T: Encode + Decode>(value: &T, name: &str) {
        assert_eq!(encode(value).len(), T::MIN_SIZE, "{name}");
    }
    let zero = [0u8; 32];
    let id = [0u8; 16];
    let statement = PublicStatement {
        claim_list_digest: zero,
        weights_root: zero,
        weights_epoch: 0,
        manifest_digest: zero,
        batch: BatchDescriptor { requests: vec![] },
        predictor: PredictorSpec::Exact {
            decoding: Decoding::Greedy,
        },
    };
    let round1 = Round1 {
        protocol_version: 0,
        session_id: id,
        statement: statement.clone(),
        advice: AdviceCommitment {
            root: zero,
            layout_digest: zero,
        },
    };
    let empty_sumcheck = SumcheckProof {
        rounds: vec![],
        final_evals: vec![],
    };
    let round2 = Round2Response {
        logup: LogUpProof {
            roots: vec![],
            values: vec![],
        },
        trees: TreeProof {
            levels: vec![],
            splits: vec![],
        },
        stages: vec![],
        advice_combined: vec![],
        weight_combined: vec![],
        scoring: ScoringClaims { requests: vec![] },
    };
    let round3 = Round3Response {
        advice_columns: vec![],
        advice_proof: MultiProof::default(),
        weight_columns: vec![],
    };
    smallest(
        &SeedCommit {
            session_id: id,
            commitment: zero,
        },
        "SeedCommit",
    );
    smallest(
        &ProverSeedShare {
            session_id: id,
            share: zero,
        },
        "ProverSeedShare",
    );
    smallest(
        &SeedReveal {
            session_id: id,
            share: zero,
        },
        "SeedReveal",
    );
    smallest(&statement, "PublicStatement");
    smallest(&statement.predictor, "PredictorSpec");
    smallest(&round1, "Round1");
    smallest(&round2, "Round2Response");
    smallest(
        &StageProof {
            stage: StageId(0),
            sumcheck: empty_sumcheck,
            splits: vec![],
        },
        "StageProof",
    );
    smallest(&FixedBits(0), "FixedBits");
    smallest(
        &ColumnQueries {
            advice: vec![],
            weights: vec![],
        },
        "ColumnQueries",
    );
    smallest(&round3, "Round3Response");
    smallest(
        &NiProof {
            round1,
            seed_nonce: 0,
            round2,
            query_nonce: 0,
            round3,
        },
        "NiProof",
    );
}
