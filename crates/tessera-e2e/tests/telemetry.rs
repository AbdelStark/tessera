// SPDX-License-Identifier: Apache-2.0 OR MIT

//! 05 §2 and §5 over a real compile, weight commitment, and NI and Live proof of the exact-cli
//! fixture: every `tessera` callsite registered on the way has a well-formed name, events
//! `tessera.<crate>.<event>` with their crate's own segment and a message template, spans one of
//! 05 §5's with their public fields, and every prover-side span of 05 §5 is reached.
//! `cargo xtask lint` checks the names of the callsites a run does not reach.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tessera_codec::{Ascii, DecodeLimits, Secret, SecretBuf, envelope, schema};
use tessera_ir::{GadgetPolicy, IntegerModel, ParamSetId};
use tessera_pcs::{WeightCommitment, WeightCommitmentState, WeightLayout, WeightMatrix};
use tessera_protocol::{self as protocol, ParamSet};
use tessera_prover::{
    Backend, MaskRng, MemoryBudget, Mode, ProofBatch, Prover, ProverConfig, ReleaseConfig,
};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::{Event, Metadata, Subscriber};

/// Records every callsite registered while it is the global subscriber.
#[derive(Clone, Default)]
struct Callsites {
    seen: Arc<Mutex<Vec<&'static Metadata<'static>>>>,
    next: Arc<AtomicU64>,
}
impl Subscriber for Callsites {
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        self.seen.lock().unwrap().push(metadata);
        Interest::always()
    }
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(self.next.fetch_add(1, Ordering::Relaxed) + 1)
    }
    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, _: &Event<'_>) {}
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

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

fn config(mode: Mode) -> ProverConfig {
    ProverConfig {
        backend: Backend::Cpu { threads: 1 },
        params: ParamSet::TEST_SMALL,
        mode,
        memory_budget: MemoryBudget { host: 64 << 30 },
        mask_rng: MaskRng::OsRng,
        release: ReleaseConfig,
    }
}

/// Compiles, commits and proves once non-interactively and once live, with fresh randomness.
fn run() {
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
    Prover::new(
        &compiled.plan,
        &model,
        &mut state,
        config(Mode::NonInteractive),
    )
    .unwrap()
    .prove_non_interactive(&batch)
    .unwrap();

    let mut state = WeightCommitmentState::new(public).unwrap();
    let mut live = Prover::new(&compiled.plan, &model, &mut state, config(Mode::Live)).unwrap();
    let (share, session_id) = ([5; 32], [6; 16]);
    live.start(
        &batch,
        &protocol::SeedCommit {
            session_id,
            commitment: protocol::coin_commitment(&share),
        },
    )
    .unwrap();
    let round2 = live
        .on_seed(&protocol::SeedReveal { session_id, share })
        .unwrap();
    let draws = [0u64, 3, 5];
    let queries = protocol::ColumnQueries {
        advice: (0..params.advice_queries)
            .map(|i| u32::try_from(draws[i as usize % 3]).unwrap())
            .collect(),
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
                    indices: (0..params.weight_queries)
                        .map(|i| u32::try_from(draws[i as usize % 3] % (1 << log2_n)).unwrap())
                        .collect(),
                }
            })
            .collect(),
    };
    live.on_queries(&queries).unwrap();
}

fn snake(text: &str) -> bool {
    !text.is_empty()
        && text.starts_with(|c: char| c.is_ascii_lowercase())
        && text.split('_').all(|p| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        })
}

#[test]
fn every_registered_callsite_follows_05() {
    let callsites = Callsites::default();
    tracing::subscriber::set_global_default(callsites.clone()).unwrap();
    run();

    // 05 §5's spans and their public fields; the verifier's are its own.
    let spans: BTreeMap<&str, &[&str]> = BTreeMap::from([
        ("compile", &["params", "layers"][..]),
        ("commit_weights", &["epoch"][..]),
        ("prove", &["batch"][..]),
        ("witness_pass", &["layer_tile"][..]),
        ("round1", &[][..]),
        ("stage", &["id"][..]),
        ("opening", &["commitment"][..]),
        ("round3", &[][..]),
    ]);
    let seen = callsites.seen.lock().unwrap().clone();
    let mut reached = BTreeSet::new();
    let mut events = BTreeSet::new();
    let mut wrong = Vec::new();
    for metadata in seen.iter().filter(|m| m.target().starts_with("tessera_")) {
        let fields: Vec<&str> = metadata.fields().iter().map(|f| f.name()).collect();
        let krate = metadata.target().split("::").next().unwrap();
        let short = krate.strip_prefix("tessera_").unwrap();
        let name = metadata.name();
        if metadata.is_span() {
            match spans.get(name) {
                Some(expected) if fields == *expected => {
                    reached.insert(name);
                }
                _ => wrong.push(format!("span {name}{fields:?} at {}", metadata.target())),
            }
        } else {
            let parts: Vec<&str> = name.split('.').collect();
            let well_formed = parts.len() == 3
                && parts[0] == "tessera"
                && parts[1] == short
                && snake(parts[2])
                && fields.last() == Some(&"message")
                && fields.iter().all(|f| snake(f));
            if well_formed {
                events.insert(name);
            } else {
                wrong.push(format!("event {name}{fields:?} at {}", metadata.target()));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
    let expected: BTreeSet<&str> = spans.keys().copied().collect();
    assert_eq!(reached, expected, "every prover-side span of 05 §5");
    for event in [
        "tessera.compiler.compiled",
        "tessera.pcs.weights_committed",
        "tessera.pcs.row_block_committed",
        "tessera.prover.witness_computed",
        "tessera.prover.round1_committed",
        "tessera.prover.stage_proved",
        "tessera.prover.opening_combined",
        "tessera.prover.round3_answered",
        "tessera.prover.proof_completed",
    ] {
        assert!(events.contains(event), "{event} registered: {events:?}");
    }
}
