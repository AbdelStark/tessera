// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054 'Pass structure' and RFC-0018 'Pass counters' over a real CPU proof: a leaf stage
//! that fits counts one P2 and no P2b; one that a forced budget groups counts one P2, the group
//! pass, and one P2b, the merge's pass, whatever its groups. This binary owns the process-wide
//! metrics registry, which no other test in it touches.

use std::fs::{self, File};
use std::io::Read;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tessera_codec::metrics::{self, Pass, Sample};
use tessera_codec::{Ascii, DecodeLimits, Secret, SecretBuf, envelope, schema};
use tessera_ir::{GadgetPolicy, IntegerModel, ParamSetId};
use tessera_pcs::{WeightCommitment, WeightCommitmentState, WeightLayout};
use tessera_protocol::{self as protocol, ParamSet};
use tessera_prover::{
    Backend, MaskRng, MemoryBudget, Mode, ProofBatch, Prover, ProverConfig, ReleaseConfig,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The exact-cli fixture: a tiny three-layer model and its recorded request of five tokens.
fn fixture() -> (IntegerModel, Value) {
    let dir = root().join("spec/vectors/exact-cli");
    let mut file = File::open(dir.join("tiny-3-reference.d")).unwrap();
    let len = usize::try_from(file.metadata().unwrap().len()).unwrap();
    let mut bytes = SecretBuf::try_with_capacity(len).unwrap();
    bytes.resize(len, 0);
    file.read_exact(&mut bytes).unwrap();
    let model =
        envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL).unwrap();
    let meta = serde_json::from_slice(&fs::read(dir.join("expected.json")).unwrap()).unwrap();
    (model, meta)
}

fn config(host: u64) -> ProverConfig {
    ProverConfig {
        backend: Backend::Cpu { threads: 1 },
        params: ParamSet::TEST_SMALL,
        mode: Mode::NonInteractive,
        memory_budget: MemoryBudget { host },
        mask_rng: MaskRng::OsRng,
        release: ReleaseConfig,
    }
}

/// `tessera_passes_total{pass}` in `samples`, 0 when the pass never ran.
fn passes(samples: &[Sample], pass: &str) -> u64 {
    samples
        .iter()
        .find(|s| {
            s.name == "tessera_passes_total"
                && s.labels.iter().any(|(k, v)| *k == "pass" && v == pass)
        })
        .map_or(0, |s| match s.value {
            metrics::Value::Counter(n) => n,
            _ => panic!("tessera_passes_total is a counter"),
        })
}

#[test]
fn a_grouped_proof_counts_the_group_pass_and_the_merge_pass_once_each() {
    let (model, meta) = fixture();
    let params = ParamSet::TEST_SMALL;
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
    let hex = |key: &str| {
        let text = meta[key].as_str().unwrap();
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
            .collect::<Vec<u8>>()
    };
    let batch = ProofBatch {
        statement: protocol::PublicStatement {
            claim_list_digest: tessera_hash::hash(
                tessera_hash::Domain::ClaimList,
                &tessera_codec::encode(&compiled.claim_list),
            ),
            weights_root: commitment.weights_root(),
            weights_epoch: 0,
            manifest_digest: hex("manifest_digest").try_into().unwrap(),
            batch: protocol::BatchDescriptor {
                requests: vec![protocol::RequestRef {
                    request_id: meta["request_id"].as_u64().unwrap(),
                    session_id: hex("request_session").try_into().unwrap(),
                    prompt_len: 2,
                    output_len: 3,
                }],
            },
            predictor: protocol::PredictorSpec::Exact {
                decoding: protocol::Decoding::Greedy,
            },
        },
        tokens: Secret::new(
            meta["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| u32::try_from(t.as_u64().unwrap()).unwrap())
                .collect(),
        ),
    };
    let public = *commitment.public();
    let mut state = WeightCommitmentState::new(public).unwrap();
    let prove = |state: &mut WeightCommitmentState, host: u64| {
        metrics::reset();
        let proof = Prover::new(&compiled.plan, &model, state, config(host))
            .unwrap()
            .prove_non_interactive(&batch)
            .unwrap();
        (proof, metrics::snapshot())
    };

    // A stage that fits: one P2, no merge pass.
    let fits = Prover::new(&compiled.plan, &model, &mut state, config(64 << 30))
        .unwrap()
        .plan(&batch)
        .unwrap();
    assert!(!fits.passes().contains(&Pass::P2b));
    let (proof, samples) = prove(&mut state, 64 << 30);
    assert_eq!(proof.round1.leaf_groups, protocol::LeafGroups::NONE);
    assert_eq!((passes(&samples, "p2"), passes(&samples, "p2b")), (1, 0));

    // One byte short of it: the stage in groups, one group pass and one merge pass.
    let budget = fits.resident_bytes().unwrap() - 1;
    let grouped = Prover::new(&compiled.plan, &model, &mut state, config(budget))
        .unwrap()
        .plan(&batch)
        .unwrap();
    assert_ne!(grouped.leaf_groups(), protocol::LeafGroups::NONE);
    let (proof, samples) = prove(&mut state, budget);
    assert_eq!(proof.round1.leaf_groups, grouped.leaf_groups());
    assert!(proof.round2.trees.groups.len() >= 2);
    assert_eq!((passes(&samples, "p2"), passes(&samples, "p2b")), (1, 1));
}
