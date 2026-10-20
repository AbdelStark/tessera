// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What the debug forms and errors show (05 §3, 04 §1 principle 4): round messages,
//! evaluations, combined rows, opened columns and multiproofs are protocol-class, so their
//! debug forms show lengths only; the verifier's share stays out of its session's debug form
//! until `SeedReveal` carries it; and no session shows a share or a commitment.
//!
//! Each message is filled with a canary, a field element or a byte, and its debug form must
//! hold the canary in none of the forms a value prints in: decimal, and hexadecimal in either
//! case.

mod common;

use common::{Checks, Engine, SplitMix64, advice, statement};
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_protocol::{
    AdviceCommitment, ColumnQueries, CombinedRow, LogUpProof, MatrixColumns, MatrixQueries,
    MultiProof, NiProof, PROTOCOL_VERSION, ProverSeedShare, ProverSession, Round1, Round2Response,
    Round3Response, ScoringClaims, SeedCommit, StageId, StageProof, SumcheckProof, TreeProof,
    VerifierSession, WeightCombinedRows, coin_commitment,
};

/// An element no length or index in these messages comes near.
const CANARY: u64 = 0x0123_4567_89AB_CDEF;

/// A byte whose decimal form, 171, no count here reaches.
const BYTE: u8 = 0xAB;

/// The forms `x` prints in.
fn forms(x: u64) -> [String; 3] {
    [x.to_string(), format!("{x:x}"), format!("{x:X}")]
}

/// Fails if `text` holds the canary element or the canary byte in any form.
fn assert_redacted(what: &str, text: &str) {
    for form in forms(CANARY) {
        assert!(!text.contains(&form), "{what} shows {form}: {text}");
    }
    let byte = u64::from(BYTE);
    for form in forms(byte) {
        assert!(
            !text.contains(&format!("{form}, {form}")),
            "{what} shows bytes: {text}"
        );
    }
    assert!(
        !text.to_lowercase().contains("abab"),
        "{what} shows bytes: {text}"
    );
}

fn canary_fp() -> Goldilocks {
    Goldilocks::from_canonical_u64(CANARY).expect("below P")
}

fn canary() -> GoldilocksExt2 {
    GoldilocksExt2::new(canary_fp(), canary_fp())
}

fn row() -> CombinedRow {
    CombinedRow {
        u: vec![canary(); 3],
        mask: vec![canary(); 2],
    }
}

fn round2() -> Round2Response {
    Round2Response {
        logup: LogUpProof {
            roots: vec![canary(); 4],
            values: vec![canary()],
        },
        trees: TreeProof {
            levels: vec![SumcheckProof {
                rounds: vec![vec![canary(); 3]],
                final_evals: vec![canary(); 4],
            }],
            splits: vec![canary(); 2],
        },
        stages: vec![StageProof {
            stage: StageId(1),
            sumcheck: SumcheckProof {
                rounds: vec![vec![canary(); 2]],
                final_evals: vec![canary()],
            },
            splits: vec![canary()],
        }],
        advice_combined: vec![row(), row()],
        weight_combined: vec![WeightCombinedRows {
            matrix: 1,
            evaluations: vec![row()],
            proximity: row(),
        }],
        scoring: ScoringClaims { requests: vec![] },
    }
}

fn round3() -> Round3Response {
    let proof = MultiProof {
        siblings: vec![[BYTE; 32]; 2],
    };
    Round3Response {
        advice_columns: vec![vec![canary_fp(); 3]; 2],
        advice_proof: proof.clone(),
        weight_columns: vec![MatrixColumns {
            matrix: 8,
            columns: vec![vec![canary_fp(); 2]],
            proof,
        }],
    }
}

#[test]
fn protocol_class_messages_show_lengths_only() {
    let queries = ColumnQueries {
        advice: vec![CANARY as u32; 3],
        weights: vec![MatrixQueries {
            matrix: 1,
            indices: vec![CANARY as u32; 2],
        }],
    };
    // Public parts of fixed bytes, so that the canary byte cannot appear in them by chance.
    let mut rng = SplitMix64::new(0x98_3001);
    let mut public = statement(&mut rng);
    public.claim_list_digest = [1; 32];
    public.weights_root = [2; 32];
    public.manifest_digest = [3; 32];
    for request in &mut public.batch.requests {
        request.session_id = [4; 16];
    }
    let ni = NiProof {
        round1: Round1 {
            protocol_version: PROTOCOL_VERSION,
            session_id: [5; 16],
            statement: public,
            advice: AdviceCommitment {
                root: [6; 32],
                layout_digest: [7; 32],
            },
        },
        seed_nonce: 1,
        round2: round2(),
        query_nonce: 2,
        round3: round3(),
    };
    for (what, text) in [
        ("Round2Response", format!("{:?}", round2())),
        ("CombinedRow", format!("{:?}", row())),
        ("Round3Response", format!("{:?}", round3())),
        ("ColumnQueries", format!("{queries:?}")),
        ("NiProof", format!("{ni:?}")),
    ] {
        assert_redacted(what, &text);
    }
    // The 32-bit canary's own forms, for the queries' indices.
    let index = u64::from(CANARY as u32);
    let text = format!("{queries:?}");
    for form in forms(index) {
        assert!(!text.contains(&form), "the queries show {form}: {text}");
    }
    // The lengths are there.
    assert!(format!("{:?}", row()).contains("u: 3"));
}

/// Before the reveal, no debug form of the verifier's session holds its share; the prover's
/// session shows neither share nor commitment.
#[test]
fn sessions_show_no_share() {
    let mut verifier = VerifierSession::new(Checks::new());
    let commit = verifier.start().expect("SeedCommit");
    let before = format!("{verifier:?}");
    let mut rng = SplitMix64::new(0x98_3002);
    let round1 = Round1 {
        protocol_version: PROTOCOL_VERSION,
        session_id: commit.session_id,
        statement: statement(&mut rng),
        advice: advice(&mut rng),
    };
    let share = ProverSeedShare {
        session_id: commit.session_id,
        share: rng.digest(),
    };
    let reveal = verifier.on_round1(&round1, &share).expect("SeedReveal");
    let s_v = reveal.share;
    assert_eq!(coin_commitment(&s_v), commit.commitment);
    for text in [before, format!("{verifier:?}")] {
        assert!(!text.contains(&common::hex(&s_v)), "{text}");
        assert!(!text.contains(&format!("{s_v:?}")), "{text}");
    }

    let mut prover = ProverSession::new(Engine::new(0x98_3003));
    let seed_commit = SeedCommit {
        session_id: [9; 16],
        commitment: coin_commitment(&[BYTE; 32]),
    };
    let (_, share) = prover.start(&seed_commit).expect("Round1");
    let text = format!("{prover:?}");
    for secret in [share.share, seed_commit.commitment] {
        assert!(!text.contains(&common::hex(&secret)), "{text}");
        assert!(!text.contains(&format!("{secret:?}")), "{text}");
    }
}
