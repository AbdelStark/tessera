// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The invariants of RFC-0003 over every message of RFC-0036, for random values:
//!
//! - INV-TCE-1: `decode(encode(m)) = m`.
//! - INV-TCE-2: whatever decodes re-encodes to the same bytes. The inputs are valid encodings
//!   after up to four edits (a byte replaced, inserted or deleted), which explore the boundary
//!   between canonical and not. A value an edited input decodes to is also judged without the
//!   decoder: its matrix ids name matrices and strictly increase, and its rationals are
//!   reduced with a positive denominator.
//! - INV-TCE-4: a trailing byte is refused.
//! - Every encoding is at least `MIN_SIZE` bytes, the bound the length checks rely on.

use std::fmt::Debug;

use proptest::prelude::*;
use tessera_codec::error::ErrorCode;
use tessera_codec::{Decode, DecodeLimits, Encode, Rational, decode, encode};
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_protocol::{
    AdviceCommitment, BatchDescriptor, ColumnQueries, CombinedRow, Decoding, FixedBits, GroupProof,
    LeafGroups, LogUpProof, MAX_LEAF_GROUP_BITS, MatrixColumns, MatrixQueries, MultiProof, NiProof,
    PredictorSpec, ProverSeedShare, PublicStatement, RequestRef, Round1, Round2Response,
    Round3Response, ScoringClaims, SeedCommit, SeedReveal, StageId, StageProof, SumcheckProof,
    TreeProof, WeightCombinedRows, WeightMatrix,
};

const LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 1 << 20,
    max_vec_len: 1 << 16,
    max_depth: 64,
};

/// Up to four edits of an encoding: `(kind, position, byte)`, where the kind replaces, inserts
/// or deletes.
type Edits = Vec<(u8, usize, u8)>;

fn edits() -> impl Strategy<Value = Edits> {
    prop::collection::vec((0..3u8, any::<usize>(), any::<u8>()), 1..=4)
}

fn apply(mut bytes: Vec<u8>, edits: &Edits) -> Vec<u8> {
    for &(kind, position, byte) in edits {
        let at = position % (bytes.len() + 1);
        match kind {
            0 if at < bytes.len() => bytes[at] = byte,
            1 => bytes.insert(at, byte),
            _ if at < bytes.len() => {
                bytes.remove(at);
            }
            _ => bytes.push(byte),
        }
    }
    bytes
}

/// Every invariant for `value`, `canonical` judging what an edited encoding decodes to.
fn laws<T: Encode + Decode + PartialEq + Debug>(
    value: T,
    edits: &Edits,
    canonical: impl Fn(&T) -> bool,
) -> Result<(), TestCaseError> {
    let encoded = encode(&value);
    prop_assert!(encoded.len() >= T::MIN_SIZE);
    prop_assert!(canonical(&value));
    prop_assert_eq!(decode::<T>(&encoded, LIMITS), Ok(value));
    let edited = apply(encoded.clone(), edits);
    if let Ok(other) = decode::<T>(&edited, LIMITS) {
        prop_assert!(canonical(&other), "accepted {:?}", other);
        prop_assert_eq!(encode(&other), edited);
    }
    let mut longer = encoded;
    longer.push(0);
    prop_assert_eq!(
        decode::<T>(&longer, LIMITS).map(drop).map_err(|e| e.code),
        Err(ErrorCode::CodecNoncanonical)
    );
    Ok(())
}

// ----- The judges: what a canonical value is, without the decoder ---------------------------------

/// Matrix ids that name a matrix, strictly increasing.
fn keyed(ids: impl Iterator<Item = u8>) -> bool {
    let ids: Vec<u8> = ids.collect();
    ids.iter()
        .all(|&id| usize::from(id) < WeightMatrix::ALL.len())
        && ids.windows(2).all(|pair| pair[0] < pair[1])
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn reduced(decoding: &Decoding) -> bool {
    match decoding {
        Decoding::Greedy => true,
        Decoding::Temperature { t } => t.den() > 0 && gcd(t.num().unsigned_abs(), t.den()) == 1,
    }
}

fn round1_canonical(round1: &Round1) -> bool {
    match &round1.statement.predictor {
        PredictorSpec::Exact { decoding } | PredictorSpec::Mixture { decoding, .. } => {
            reduced(decoding)
        }
    }
}

fn round2_canonical(round2: &Round2Response) -> bool {
    keyed(round2.weight_combined.iter().map(|w| w.matrix))
}

fn queries_canonical(queries: &ColumnQueries) -> bool {
    keyed(queries.weights.iter().map(|w| w.matrix))
}

fn round3_canonical(round3: &Round3Response) -> bool {
    keyed(round3.weight_columns.iter().map(|w| w.matrix))
}

fn any_value<T>(_: &T) -> bool {
    true
}

// ----- Strategies ---------------------------------------------------------------------------------

const P: u64 = Goldilocks::MODULUS;

fn fp() -> impl Strategy<Value = Goldilocks> {
    prop_oneof![Just(0), Just(P - 1), 0..P,]
        .prop_map(|x| Goldilocks::from_canonical_u64(x).expect("below P"))
}

fn fp2() -> impl Strategy<Value = GoldilocksExt2> {
    (fp(), fp()).prop_map(|(c0, c1)| GoldilocksExt2::new(c0, c1))
}

fn fp2s() -> impl Strategy<Value = Vec<GoldilocksExt2>> {
    prop::collection::vec(fp2(), 0..4)
}

/// Sorted distinct matrix ids, at least `least` and at most 4 of them.
fn matrices(least: usize) -> impl Strategy<Value = Vec<u8>> {
    let ids: Vec<u8> = (0..WeightMatrix::ALL.len() as u8).collect();
    prop::sample::subsequence(ids, least..=4)
}

fn decoding() -> impl Strategy<Value = Decoding> {
    prop_oneof![
        Just(Decoding::Greedy),
        (any::<i64>(), 1..=u64::MAX).prop_map(|(num, den)| Decoding::Temperature {
            t: Rational::new(num, den).expect("a positive denominator"),
        }),
    ]
}

fn predictor() -> impl Strategy<Value = PredictorSpec> {
    prop_oneof![
        decoding().prop_map(|decoding| PredictorSpec::Exact { decoding }),
        (decoding(), any::<u8>())
            .prop_map(|(decoding, eps_log2)| PredictorSpec::Mixture { decoding, eps_log2 }),
    ]
}

fn statement() -> impl Strategy<Value = PublicStatement> {
    let request = (any::<u64>(), any::<[u8; 16]>(), any::<u32>(), any::<u32>()).prop_map(
        |(request_id, session_id, prompt_len, output_len)| RequestRef {
            request_id,
            session_id,
            prompt_len,
            output_len,
        },
    );
    (
        any::<[u8; 32]>(),
        any::<[u8; 32]>(),
        any::<u32>(),
        any::<[u8; 32]>(),
        prop::collection::vec(request, 0..4),
        predictor(),
    )
        .prop_map(
            |(
                claim_list_digest,
                weights_root,
                weights_epoch,
                manifest_digest,
                requests,
                predictor,
            )| {
                PublicStatement {
                    claim_list_digest,
                    weights_root,
                    weights_epoch,
                    manifest_digest,
                    batch: BatchDescriptor { requests },
                    predictor,
                }
            },
        )
}

fn round1() -> impl Strategy<Value = Round1> {
    (
        any::<u16>(),
        any::<[u8; 16]>(),
        statement(),
        any::<[u8; 32]>(),
        any::<[u8; 32]>(),
        leaf_groups(),
    )
        .prop_map(
            |(protocol_version, session_id, statement, root, layout_digest, leaf_groups)| Round1 {
                protocol_version,
                session_id,
                statement,
                advice: AdviceCommitment {
                    root,
                    layout_digest,
                },
                leaf_groups,
            },
        )
}

/// Every grouping RFC-0054 allows, `token_bits + layer_bits ≤ 14`.
fn leaf_groups() -> impl Strategy<Value = LeafGroups> {
    (0..=MAX_LEAF_GROUP_BITS)
        .prop_flat_map(|g| (0..=g).prop_map(move |t| (t, g - t)))
        .prop_map(|(token_bits, layer_bits)| LeafGroups {
            token_bits,
            layer_bits,
        })
}

fn group() -> impl Strategy<Value = GroupProof> {
    (
        fp2s(),
        prop::collection::vec(sumcheck(), 0..3),
        fp2s(),
        fp2s(),
    )
        .prop_map(|(roots, levels, values, splits)| GroupProof {
            roots,
            levels,
            values,
            splits,
        })
}

fn sumcheck() -> impl Strategy<Value = SumcheckProof<GoldilocksExt2>> {
    (prop::collection::vec(fp2s(), 0..3), fp2s()).prop_map(|(rounds, final_evals)| SumcheckProof {
        rounds,
        final_evals,
    })
}

fn row() -> impl Strategy<Value = CombinedRow> {
    (fp2s(), fp2s()).prop_map(|(u, mask)| CombinedRow { u, mask })
}

fn round2() -> impl Strategy<Value = Round2Response> {
    opening(0)
}

/// A `Round2Response` that opens at least `least` matrices.
fn opening(least: usize) -> impl Strategy<Value = Round2Response> {
    let stage =
        (any::<u16>(), sumcheck(), fp2s()).prop_map(|(stage, sumcheck, splits)| StageProof {
            stage: StageId(stage),
            sumcheck,
            splits,
        });
    let weights = matrices(least).prop_flat_map(|ids| {
        let n = ids.len();
        (
            Just(ids),
            prop::collection::vec((prop::collection::vec(row(), 1..3), row()), n),
        )
            .prop_map(|(ids, rows)| {
                ids.into_iter()
                    .zip(rows)
                    .map(|(matrix, (evaluations, proximity))| WeightCombinedRows {
                        matrix,
                        evaluations,
                        proximity,
                    })
                    .collect::<Vec<_>>()
            })
    });
    (
        (fp2s(), fp2s()),
        (
            prop::collection::vec(sumcheck(), 0..3),
            prop::collection::vec(group(), 0..3),
            fp2s(),
            sumcheck(),
            fp2s(),
        ),
        prop::collection::vec(stage, 0..3),
        prop::collection::vec(row(), 0..3),
        weights,
        prop::collection::vec(any::<u64>().prop_map(FixedBits), 0..4),
    )
        .prop_map(
            |(
                (roots, values),
                (levels, groups, table_roots, merge, splits),
                stages,
                advice_combined,
                weight_combined,
                requests,
            )| {
                Round2Response {
                    logup: LogUpProof { roots, values },
                    trees: TreeProof {
                        levels,
                        groups,
                        table_roots,
                        merge,
                        splits,
                    },
                    stages,
                    advice_combined,
                    weight_combined,
                    scoring: ScoringClaims { requests },
                }
            },
        )
}

fn queries() -> impl Strategy<Value = ColumnQueries> {
    let indices = || prop::collection::vec(any::<u32>(), 0..6);
    let weights = matrices(0).prop_flat_map(move |ids| {
        let n = ids.len();
        (Just(ids), prop::collection::vec(indices(), n)).prop_map(|(ids, lists)| {
            ids.into_iter()
                .zip(lists)
                .map(|(matrix, indices)| MatrixQueries { matrix, indices })
                .collect::<Vec<_>>()
        })
    });
    (indices(), weights).prop_map(|(advice, weights)| ColumnQueries { advice, weights })
}

fn multiproof() -> impl Strategy<Value = MultiProof> {
    prop::collection::vec(any::<[u8; 32]>(), 0..3).prop_map(|siblings| MultiProof { siblings })
}

fn round3() -> impl Strategy<Value = Round3Response> {
    let columns = || prop::collection::vec(prop::collection::vec(fp(), 0..3), 0..3);
    let weights = matrices(0).prop_flat_map(move |ids| {
        let n = ids.len();
        (
            Just(ids),
            prop::collection::vec((columns(), multiproof()), n),
        )
            .prop_map(|(ids, parts)| {
                ids.into_iter()
                    .zip(parts)
                    .map(|(matrix, (columns, proof))| MatrixColumns {
                        matrix,
                        columns,
                        proof,
                    })
                    .collect::<Vec<_>>()
            })
    });
    (columns(), multiproof(), weights).prop_map(|(advice_columns, advice_proof, weight_columns)| {
        Round3Response {
            advice_columns,
            advice_proof,
            weight_columns,
        }
    })
}

proptest! {
    #[test]
    fn seed_messages(
        session_id in any::<[u8; 16]>(),
        bytes in any::<[u8; 32]>(),
        edits in edits(),
    ) {
        laws(SeedCommit { session_id, commitment: bytes }, &edits, any_value)?;
        laws(ProverSeedShare { session_id, share: bytes }, &edits, any_value)?;
        laws(SeedReveal { session_id, share: bytes }, &edits, any_value)?;
    }

    #[test]
    fn round1_messages(round1 in round1(), edits in edits()) {
        laws(round1, &edits, round1_canonical)?;
    }

    #[test]
    fn round2_messages(round2 in round2(), edits in edits()) {
        laws(round2, &edits, round2_canonical)?;
    }

    #[test]
    fn column_queries(queries in queries(), edits in edits()) {
        laws(queries, &edits, queries_canonical)?;
    }

    #[test]
    fn round3_messages(round3 in round3(), edits in edits()) {
        laws(round3, &edits, round3_canonical)?;
    }

    #[test]
    fn ni_proofs(
        round1 in round1(),
        seed_nonce in any::<u64>(),
        round2 in round2(),
        query_nonce in any::<u64>(),
        round3 in round3(),
        edits in edits(),
    ) {
        let proof = NiProof { round1, seed_nonce, round2, query_nonce, round3 };
        laws(proof, &edits, |p| {
            round1_canonical(&p.round1) && round2_canonical(&p.round2) && round3_canonical(&p.round3)
        })?;
    }

    /// An entry's id is refused unless it names a matrix above the previous entry's, whatever
    /// the rest of the message: a replaced id byte gives exactly the refusal the judge predicts.
    #[test]
    fn a_replaced_matrix_id_is_judged_like_the_decoder(
        round2 in opening(1),
        entry in any::<prop::sample::Index>(),
        id in any::<u8>(),
    ) {
        let entry = entry.index(round2.weight_combined.len());
        let mut changed = round2.clone();
        changed.weight_combined[entry].matrix = id;
        let bytes = encode(&changed);
        let outcome = decode::<Round2Response>(&bytes, LIMITS);
        let ids: Vec<u8> = changed.weight_combined.iter().map(|w| w.matrix).collect();
        let first_bad = (0..ids.len()).find(|&i| !keyed(ids[..=i].iter().copied()));
        match first_bad {
            None => prop_assert_eq!(outcome, Ok(changed)),
            Some(bad) => {
                // An entry's first byte is its id: the offset is where the bad entry starts,
                // which a message holding only the entries before it ends at, its scoring aside.
                let before = Round2Response {
                    weight_combined: changed.weight_combined[..bad].to_vec(),
                    ..changed.clone()
                };
                let start = encode(&before).len() - encode(&before.scoring).len();
                let error = outcome.expect_err("refused");
                prop_assert_eq!(error.code, ErrorCode::CodecNoncanonical);
                prop_assert_eq!(error.offset, start as u64);
            }
        }
    }
}
