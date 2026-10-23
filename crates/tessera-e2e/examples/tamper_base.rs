// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writes the valid base proofs of the SC-4 tamper corpus (07 §4) to `spec/vectors/tamper/`:
//! a non-interactive proof and a recorded live session of the exact-cli fixture model, with the
//! claim list, the public weight artifact and the corpus metadata.
//!
//! The bases are synthetic producer output, inputs to the corpus and not evidence: the
//! independent `spec/vectors/tamper/generate.py` chooses every mutation and predicts its verdict.
//! `--features tamper` seeds the masks and the weight commitment (`test-rng`), which release
//! builds refuse, so the bases regenerate byte for byte.
//!
//! ```sh
//! cargo run -p tessera-e2e --features tamper --example tamper-base [-- --check | --out DIR]
//! python3 spec/vectors/tamper/generate.py
//! ```

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use tessera_codec::{Ascii, DecodeLimits, Secret, SecretBuf, envelope, schema};
use tessera_hash::{Domain, ParamSetId, SessionContext, Transcript};
use tessera_ir::{GadgetPolicy, IntegerModel};
use tessera_pcs::{WeightCommitment, WeightCommitmentState, WeightLayout, WeightMatrix};
use tessera_protocol::{self as protocol, ParamSet};
use tessera_prover::{
    Backend, LeafLayout, MaskRng, MemoryBudget, Mode, ProofBatch, Prover, ProverConfig,
    ReleaseConfig,
};

/// Fixed live draws, repeats included, reduced into each matrix's codeword. Each holds 0, the
/// smallest index the column checks report.
const QUERIES: [u64; 8] = [0, 1, 1, 3, 5, 17, 31, 63];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex<const N: usize>(text: &str) -> [u8; N] {
    let bytes: Vec<u8> = (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect();
    bytes.try_into().unwrap()
}

/// The exact-cli fixture: its synthetic model, read straight into clearing storage (OBS-6), and
/// its public request metadata and reference tokens from the independent Python oracle.
fn fixture() -> (IntegerModel, Value) {
    let dir = root().join("spec/vectors/exact-cli");
    let mut file = File::open(dir.join("tiny-3-reference.d")).unwrap();
    let len = file.metadata().unwrap().len();
    assert!(len <= DecodeLimits::INTEGER_MODEL.max_total_bytes);
    let mut bytes = SecretBuf::try_with_capacity(usize::try_from(len).unwrap()).unwrap();
    bytes.resize(usize::try_from(len).unwrap(), 0);
    file.read_exact(&mut bytes).unwrap();
    let model =
        envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL).unwrap();
    let meta = serde_json::from_slice(&fs::read(dir.join("expected.json")).unwrap()).unwrap();
    (model, meta)
}

fn config(mode: Mode) -> ProverConfig {
    ProverConfig {
        backend: Backend::Cpu { threads: 1 },
        params: ParamSet::TEST_SMALL,
        mode,
        memory_budget: MemoryBudget { host: 64 << 30 },
        mask_rng: MaskRng::Seeded(0x135),
        release: ReleaseConfig,
    }
}

/// The real 20-bit predicates of `ni-100` on a fixed context, for both phases: the first valid
/// nonce, and the next nonce that fails. Test-small grinds no bits, so its bases cannot.
fn pow() -> Value {
    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("ni-100/1").unwrap()),
        statement_digest: [7; 32],
        session_id: [9; 16],
    };
    let cases: Vec<Value> = [("seed", "seed-pow"), ("queries", "query-pow")]
        .into_iter()
        .map(|(phase, label)| {
            let valid = Transcript::new(&context).grind(label, 20);
            let invalid = (valid + 1..)
                .find(|&n| !Transcript::new(&context).check_grind(label, 20, n))
                .unwrap();
            json!({"phase": phase, "valid": valid, "invalid": invalid})
        })
        .collect();
    json!({
        "params": "ni-100/1",
        "bits": 20,
        "statement_digest": hex(&context.statement_digest),
        "session_id": hex(&context.session_id),
        "cases": cases,
    })
}

/// Every file of the corpus's base, by name.
fn artifacts() -> BTreeMap<String, Vec<u8>> {
    let (model, meta) = fixture();
    let number = |key: &str| meta[key].as_u64().unwrap();
    let prompt_len = u32::try_from(number("prompt_len")).unwrap();
    let output_len = u32::try_from(number("output_len")).unwrap();
    let params = ParamSet::TEST_SMALL;
    // The exact-cli suite's gadget policy (docs/testing/exact-cli.md).
    let policy = GadgetPolicy {
        max_log2_rows: 18,
        query_block: 2,
        ..Default::default()
    };
    let compiled = tessera_compiler::compile(
        &model,
        &policy,
        params.layout,
        ParamSetId(Ascii::new(params.id).unwrap()),
    )
    .unwrap();
    let layout = WeightLayout::new(params.layout.weights, model.weight_shapes().unwrap()).unwrap();
    let commitment = WeightCommitment::commit(
        &layout,
        NonZeroU32::new(params.row_block()).unwrap(),
        0,
        &model.model_weights().unwrap(),
    )
    .unwrap();
    let weights = *commitment.public();
    let statement = protocol::PublicStatement {
        claim_list_digest: tessera_hash::hash(
            Domain::ClaimList,
            &tessera_codec::encode(&compiled.claim_list),
        ),
        weights_root: commitment.weights_root(),
        weights_epoch: 0,
        manifest_digest: unhex(meta["manifest_digest"].as_str().unwrap()),
        batch: protocol::BatchDescriptor {
            requests: vec![protocol::RequestRef {
                request_id: number("request_id"),
                session_id: unhex(meta["request_session"].as_str().unwrap()),
                prompt_len,
                output_len,
            }],
        },
        predictor: protocol::PredictorSpec::Exact {
            decoding: protocol::Decoding::Greedy,
        },
    };
    let tokens: Vec<u32> = meta["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| u32::try_from(t.as_u64().unwrap()).unwrap())
        .collect();
    // Public synthetic tokens of the fixture, never a user's.
    let batch = ProofBatch {
        statement,
        tokens: Secret::new(tokens),
    };
    let state = || {
        WeightCommitmentState::with_seed(
            weights,
            [19; 32],
            core::array::from_fn(|_| Default::default()),
        )
    };

    let mut ni_state = state();
    let proof = Prover::new(
        &compiled.plan,
        &model,
        &mut ni_state,
        config(Mode::NonInteractive),
    )
    .unwrap()
    .prove_non_interactive(&batch)
    .unwrap();

    // A recorded live session: fixed verifier share and queries, replayed only by tests.
    let mut live_state = state();
    let mut live =
        Prover::new(&compiled.plan, &model, &mut live_state, config(Mode::Live)).unwrap();
    let (verifier_share, session_id) = ([37; 32], [29; 16]);
    let commit = protocol::SeedCommit {
        session_id,
        commitment: protocol::coin_commitment(&verifier_share),
    };
    let (round1, share) = live.start(&batch, &commit).unwrap();
    let round2 = live
        .on_seed(&protocol::SeedReveal {
            session_id,
            share: verifier_share,
        })
        .unwrap();
    let queries = protocol::ColumnQueries {
        advice: QUERIES.iter().map(|&q| u32::try_from(q).unwrap()).collect(),
        weights: round2
            .weight_combined
            .iter()
            .map(|m| {
                let log2_n = layout
                    .matrix(WeightMatrix::ALL[usize::from(m.matrix)])
                    .log2_k()
                    + 2;
                protocol::MatrixQueries {
                    matrix: m.matrix,
                    indices: QUERIES
                        .iter()
                        .map(|&q| u32::try_from(q % (1 << log2_n)).unwrap())
                        .collect(),
                }
            })
            .collect(),
    };
    let round3 = live.on_queries(&queries).unwrap();

    // The lookup instances in schedule order, which the verifier re-derives and compares.
    let lowering = tessera_compiler::lower(
        tessera_compiler::PublicModel::of(&model),
        prompt_len + output_len,
    )
    .unwrap();
    let ranges = tessera_compiler::analyze(&lowering.graph, &lowering.tables).unwrap();
    let selected = tessera_compiler::select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &compiled.plan.claims.gadget_policy,
    )
    .unwrap();
    let runtime = tessera_compiler::RuntimePlan::derive(
        &lowering,
        &selected,
        &compiled.plan.claims,
        prompt_len,
        0,
    )
    .unwrap();
    let layers = tessera_compiler::schedule::LayerMap::new(&lowering).unwrap();
    let leaf = LeafLayout::runtime(&lowering, &layers, &runtime, &compiled.plan.claims).unwrap();
    let lookup_tables: Vec<u32> = leaf.instances().iter().map(|i| i.table.0).collect();

    let base = json!({
        "format": "tessera-tamper-base/1",
        "protocol": 1,
        "params": params.id,
        "model": "../exact-cli/tiny-3-reference.d",
        "synthetic_public_data": true,
        "layers": meta["layers"],
        "positions": prompt_len + output_len,
        "request": {
            "request_id": meta["request_id"],
            "session_id": meta["request_session"],
            "prompt_len": prompt_len,
            "output_len": output_len,
            "manifest_digest": meta["manifest_digest"],
        },
        "statement_digest": hex(&batch.statement.digest()),
        "lookup_tables": lookup_tables,
        "pow": pow(),
    });
    BTreeMap::from([
        (
            "claims.tcl".into(),
            envelope::seal(&schema::CLAIM_LIST, &compiled.claim_list),
        ),
        (
            "weights.twr".into(),
            envelope::seal(&schema::WEIGHT_COMMITMENT_PUBLIC, &weights),
        ),
        ("ni.tnp".into(), envelope::seal(&schema::NI_PROOF, &proof)),
        (
            "live-round1.tce".into(),
            envelope::seal(&schema::ROUND1, &round1),
        ),
        (
            "live-round2.tce".into(),
            envelope::seal(&schema::ROUND2_RESPONSE, &round2),
        ),
        (
            "live-round3.tce".into(),
            envelope::seal(&schema::ROUND3_RESPONSE, &round3),
        ),
        (
            "live-queries.tce".into(),
            envelope::seal(&schema::COLUMN_QUERIES, &queries),
        ),
        (
            "live-seed.bin".into(),
            protocol::coin_toss(&verifier_share, &share.share).to_vec(),
        ),
        (
            "base.json".into(),
            format!("{}\n", serde_json::to_string_pretty(&base).unwrap()).into_bytes(),
        ),
    ])
}

fn main() {
    let mut out = root().join("spec/vectors/tamper");
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
            "--out" => out = PathBuf::from(args.next().expect("--out DIR")),
            _ => panic!("usage: tamper-base [--check] [--out DIR]"),
        }
    }
    let files = artifacts();
    if !check {
        fs::create_dir_all(&out).unwrap();
    }
    for (name, bytes) in files {
        let path = out.join(&name);
        if check {
            assert!(
                fs::read(&path).unwrap() == bytes,
                "{name} is stale: rerun without --check"
            );
        } else {
            fs::write(path, bytes).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    /// "Corpus regenerated deterministically" (#135): two in-process runs, then the files.
    #[test]
    fn committed_bases_regenerate_byte_for_byte() {
        let first = super::artifacts();
        assert!(first == super::artifacts(), "two runs differ");
        let dir = super::root().join("spec/vectors/tamper");
        for (name, bytes) in first {
            let committed = std::fs::read(dir.join(&name)).unwrap();
            assert!(
                committed == bytes,
                "{name} is stale: cargo run -p tessera-e2e --features tamper --example tamper-base"
            );
        }
    }
}
