// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration tests for `tessera inspect` (02 §2.2, RFC-0003 'Non-Goals', 03 §5, 05 §3).
//!
//! - **Codec and catalog vectors.** Every artifact type in `spec/vectors/codec/`
//!   (`ExecutionPlan`, `WeightCommitmentPublic`, `WeightCommitmentState`), plus `IntegerModel`,
//!   `ClaimList`, the eight protocol message schemas in `spec/vectors/protocol/protocol.json`,
//!   `TokenFile` and `LogitsFile` decodes and renders in both text and `--json` modes.
//! - **Secret redaction.** Canary weight entries, channel classes, channel multipliers, mask
//!   seeds, token ids and logit values never appear in stdout or stderr; only shapes and BLAKE3
//!   digests are printed (05 §3).
//! - **Adversarial inputs.** Malformed envelopes and payloads (including the rejected
//!   `WeightCommitmentState` cases in `spec/vectors/codec/weight-commitment.json`) exit 65 with
//!   their typed `ERR-CODEC-*` code, and unreadable paths exit 20 with `ERR-RES-IO`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
use tessera_codec::{DecodeLimits, envelope, schema};
use tessera_ir::{ChannelParams, IntegerModel, LogitsFile, TokenFile, WeightType};

fn tessera<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(args)
        .output()
        .expect("the tessera binary runs")
}

fn text(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("UTF-8")
}

fn scratch(test: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("tessera-cli-inspect-{test}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn vector_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors")
        .join(rel)
}

fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte"))
        .collect()
}

fn inspect_both(path: &Path) -> (String, Value) {
    let out_text = tessera(&["inspect".as_ref(), path.as_os_str()]);
    assert_eq!(
        out_text.status.code(),
        Some(0),
        "text inspect failed on {}: {}",
        path.display(),
        text(&out_text.stderr)
    );
    let rendered = text(&out_text.stdout).to_owned();

    let out_json = tessera(&["inspect".as_ref(), "--json".as_ref(), path.as_os_str()]);
    assert_eq!(
        out_json.status.code(),
        Some(0),
        "json inspect failed on {}: {}",
        path.display(),
        text(&out_json.stderr)
    );
    let parsed: Value = serde_json::from_str(text(&out_json.stdout)).expect("valid JSON");
    assert_eq!(parsed["command"], "inspect");
    (rendered, parsed)
}

#[test]
fn inspects_every_artifact_type_in_codec_and_catalog_vectors() {
    let dir = scratch("catalog");

    // 1. ExecutionPlan vectors in spec/vectors/codec/execution-plan.json (0x0202).
    let plans_json: Value = serde_json::from_str(
        &fs::read_to_string(vector_path("codec/execution-plan.json")).expect("execution-plan.json"),
    )
    .expect("JSON");
    for case in plans_json["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let path = dir.join(format!("{name}.tep"));
        fs::write(&path, unhex(case["hex"].as_str().expect("hex"))).expect("write plan");
        let (rendered, parsed) = inspect_both(&path);
        assert!(
            rendered.contains("schema: ExecutionPlan (0x0202)"),
            "{name}"
        );
        assert!(
            rendered.contains(case["claim_list_digest"].as_str().expect("digest")),
            "{name}"
        );
        assert_eq!(parsed["schema"], "ExecutionPlan");
        assert_eq!(parsed["class"], "Private");
        assert_eq!(
            parsed["artifact"]["claim_list_digest"],
            case["claim_list_digest"]
        );
    }

    // 2. WeightCommitmentPublic (0x0301) and WeightCommitmentState (0x0302) in
    //    spec/vectors/codec/weight-commitment.json.
    let wc_json: Value = serde_json::from_str(
        &fs::read_to_string(vector_path("codec/weight-commitment.json"))
            .expect("weight-commitment.json"),
    )
    .expect("JSON");
    for (idx, case) in wc_json["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .enumerate()
    {
        let name = case["name"].as_str().expect("name");
        let ty = case["type"].as_str().expect("type");
        let path = dir.join(format!("wc-{idx}.bin"));
        fs::write(&path, unhex(case["hex"].as_str().expect("hex"))).expect("write wc");
        let (rendered, parsed) = inspect_both(&path);
        assert!(rendered.contains(&format!("schema: {ty}")), "{name}");
        assert_eq!(parsed["schema"], ty);
        if let Some(root) = case["weights_root"].as_str() {
            assert!(rendered.contains(root), "{name}");
            assert_eq!(parsed["artifact"]["weights_root"], root);
        }
        if ty == "WeightCommitmentState" {
            assert_eq!(parsed["class"], "Secret");
            // Ensure the secret masks_seed from the vector never appears in output.
            let seed_bytes: Vec<u8> = case["value"]["masks_seed"]
                .as_array()
                .expect("masks_seed")
                .iter()
                .map(|v| v.as_str().expect("byte").parse::<u8>().expect("u8"))
                .collect();
            let seed_hex: String = seed_bytes.iter().map(|b| format!("{b:02x}")).collect();
            assert!(!rendered.contains(&seed_hex), "{name}: leaked masks_seed");
            assert!(
                !parsed.to_string().contains(&seed_hex),
                "{name}: leaked masks_seed in JSON"
            );
        }
    }

    // 3. ClaimList vectors in spec/vectors/claims/claim-lists.json (0x0201).
    let claims_json: Value = serde_json::from_str(
        &fs::read_to_string(vector_path("claims/claim-lists.json")).expect("claim-lists.json"),
    )
    .expect("JSON");
    for case in claims_json["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let path = dir.join(format!("{name}.tcl"));
        fs::write(&path, unhex(case["hex"].as_str().expect("hex"))).expect("write claims");
        let (rendered, parsed) = inspect_both(&path);
        assert!(rendered.contains("schema: ClaimList (0x0201)"), "{name}");
        assert_eq!(parsed["schema"], "ClaimList");
        assert_eq!(
            parsed["artifact"]["claim_list_digest"],
            case["digest"].as_str().expect("digest")
        );
    }

    // 4. IntegerModel vectors in spec/vectors/models/*.d (0x0101).
    for model_name in [
        "tiny-2-servable.d",
        "tiny-2-minimal.d",
        "tiny-2-affine.d",
        "tiny-4-servable.d",
    ] {
        let path = vector_path(&format!("models/{model_name}"));
        let (rendered, parsed) = inspect_both(&path);
        assert!(
            rendered.contains("schema: IntegerModel (0x0101)"),
            "{model_name}"
        );
        assert_eq!(parsed["schema"], "IntegerModel");
        assert_eq!(parsed["class"], "Secret");
    }

    // 5. Protocol message vectors in spec/vectors/protocol/protocol.json (0x0501..0x0520).
    let proto_json: Value = serde_json::from_str(
        &fs::read_to_string(vector_path("protocol/protocol.json")).expect("protocol.json"),
    )
    .expect("JSON");
    for (idx, msg) in proto_json["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .enumerate()
    {
        let ty = msg["type"].as_str().expect("type");
        let path = dir.join(format!("proto-{idx}-{ty}.bin"));
        fs::write(&path, unhex(msg["envelope"].as_str().expect("envelope"))).expect("write msg");
        let (rendered, parsed) = inspect_both(&path);
        assert!(rendered.contains(&format!("schema: {ty}")), "{ty}");
        assert_eq!(parsed["schema"], ty);
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn canary_model_tokens_and_logits_never_print_secret_values() {
    let dir = scratch("canary");

    // Load a conformance IntegerModel and plant distinct canary weights and channel parameters.
    let model_bytes =
        fs::read(vector_path("models/tiny-2-servable.d")).expect("read conformance model");
    let mut model: IntegerModel = envelope::open(
        &model_bytes,
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )
    .expect("decode IntegerModel");

    let canary_weights: [i8; 6] = [113, -97, 61, -29, 107, -71];
    for w in &mut model.weights {
        for (i, entry) in w.data.iter_mut().enumerate() {
            *entry = canary_weights[i % canary_weights.len()];
        }
    }
    model.quant.channels = vec![
        ChannelParams {
            ty: WeightType::Gate,
            layer: 0,
            class: vec![211; 16],
            multiplier: Vec::new(),
        },
        ChannelParams {
            ty: WeightType::Up,
            layer: 0,
            class: Vec::new(),
            multiplier: vec![54_321; 16],
        },
    ];

    let model_path = dir.join("canary.d");
    fs::write(&model_path, envelope::seal(&schema::INTEGER_MODEL, &model)).expect("write canary.d");
    let (model_text, model_json) = inspect_both(&model_path);
    let model_json_str = model_json.to_string();
    assert!(model_text.contains("schema: IntegerModel (0x0101)"));
    let embed = &model.weights[0];
    assert!(
        model_text.contains(&format!("Embed layer 0: {}x{}", embed.rows, embed.cols)),
        "{model_text}"
    );
    assert!(model_text.contains("Gate layer 0: channels=16 affine=false"));
    assert!(model_text.contains("Up layer 0: channels=16 affine=true"));
    for forbidden in ["113", "-97", "-29", "107", "-71", "211", "54321", "d431"] {
        assert!(
            !model_text.contains(forbidden),
            "canary {forbidden} leaked in text:\n{model_text}"
        );
        assert!(
            !model_json_str.contains(forbidden),
            "canary {forbidden} leaked in JSON:\n{model_json_str}"
        );
    }

    // Canary TokenFile (0x0601).
    let tokens = TokenFile {
        sequences: vec![
            vec![3_141_592_653, 2_718_281_828, 1_618_033_988],
            vec![1_234_567_891, 987_654_321],
        ],
    };
    let tokens_path = dir.join("canary.ttk");
    fs::write(&tokens_path, envelope::seal(&schema::TOKEN_FILE, &tokens)).expect("write .ttk");
    let (tokens_text, tokens_json) = inspect_both(&tokens_path);
    let tokens_json_str = tokens_json.to_string();
    assert!(tokens_text.contains("schema: TokenFile (0x0601)"));
    assert!(tokens_text.contains("sequences: 2"));
    assert!(tokens_text.contains("total_tokens: 5"));
    assert!(tokens_text.contains("sequence 0: len=3"));
    assert!(tokens_text.contains("sequence 1: len=2"));
    for canary_token in [
        "3141592653",
        "bb40e64d",
        "2718281828",
        "a205b064",
        "1618033988",
        "60713d44",
        "1234567891",
        "987654321",
    ] {
        assert!(
            !tokens_text.contains(canary_token),
            "token {canary_token} leaked in text:\n{tokens_text}"
        );
        assert!(
            !tokens_json_str.contains(canary_token),
            "token {canary_token} leaked in JSON:\n{tokens_json_str}"
        );
    }

    // Canary LogitsFile (0x0602).
    let logits = LogitsFile {
        vocab: 3,
        positions: vec![
            vec![-9_876_543_210_123, 7_654_321_098_765, -1_357_924_680_135],
            vec![2_468_013_579_246, -8_642_097_531_864, 5_555_555_555_555],
        ],
    };
    let logits_path = dir.join("canary.tlg");
    fs::write(&logits_path, envelope::seal(&schema::LOGITS_FILE, &logits)).expect("write .tlg");
    let (logits_text, logits_json) = inspect_both(&logits_path);
    let logits_json_str = logits_json.to_string();
    assert!(logits_text.contains("schema: LogitsFile (0x0602)"));
    assert!(logits_text.contains("vocab: 3"));
    assert!(logits_text.contains("positions: 2"));
    for canary_logit in [
        "9876543210123",
        "7654321098765",
        "1357924680135",
        "2468013579246",
        "8642097531864",
        "5555555555555",
    ] {
        assert!(
            !logits_text.contains(canary_logit),
            "logit {canary_logit} leaked in text:\n{logits_text}"
        );
        assert!(
            !logits_json_str.contains(canary_logit),
            "logit {canary_logit} leaked in JSON:\n{logits_json_str}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn rejects_malformed_envelopes_and_payloads_with_typed_codes() {
    let dir = scratch("adversarial");

    // 1. Rejected WeightCommitmentState vectors in spec/vectors/codec/weight-commitment.json.
    let wc_json: Value = serde_json::from_str(
        &fs::read_to_string(vector_path("codec/weight-commitment.json"))
            .expect("weight-commitment.json"),
    )
    .expect("JSON");
    for (idx, case) in wc_json["rejected"]
        .as_array()
        .expect("rejected")
        .iter()
        .enumerate()
    {
        let path = dir.join(format!("rejected-wc-{idx}.twc"));
        fs::write(&path, unhex(case["hex"].as_str().expect("hex"))).expect("write rejected");
        let output = tessera(&["inspect".as_ref(), "--json".as_ref(), path.as_os_str()]);
        assert_eq!(output.status.code(), Some(65));
        let err: Value = serde_json::from_str(text(&output.stdout)).expect("JSON error");
        assert_eq!(err["command"], "inspect");
        assert_eq!(err["error"]["code"], case["code"]);
    }

    // 2. Envelope-level faults (magic, unknown schema, truncated, bad digest, trailing byte).
    let valid = envelope::seal(
        &schema::TOKEN_FILE,
        &TokenFile {
            sequences: vec![vec![1, 2]],
        },
    );
    let mut bad_digest = valid.clone();
    *bad_digest.last_mut().expect("digest byte") ^= 1;
    let mut trailing = valid.clone();
    trailing.push(0);
    let mut unknown_schema = valid.clone();
    unknown_schema[6..8].copy_from_slice(&0xFFF0u16.to_le_bytes());

    let cases: [(&str, Vec<u8>, &str); 5] = [
        (
            "garbage.bin",
            b"not an envelope".to_vec(),
            "ERR-CODEC-MAGIC",
        ),
        ("unknown.bin", unknown_schema, "ERR-CODEC-MAGIC"),
        (
            "short.bin",
            valid[..valid.len() - 1].to_vec(),
            "ERR-CODEC-TRUNCATED",
        ),
        ("digest.bin", bad_digest, "ERR-CODEC-DIGEST"),
        ("trailing.bin", trailing, "ERR-CODEC-NONCANONICAL"),
    ];
    for (name, bytes, expected_code) in cases {
        let path = dir.join(name);
        fs::write(&path, bytes).expect("write case");
        let output = tessera(&["inspect".as_ref(), "--json".as_ref(), path.as_os_str()]);
        assert_eq!(output.status.code(), Some(65), "{name}");
        let err: Value = serde_json::from_str(text(&output.stdout)).expect("JSON error");
        assert_eq!(err["command"], "inspect", "{name}");
        assert_eq!(err["error"]["code"], expected_code, "{name}");
    }

    // 3. Unreadable file -> exit 20 ERR-RES-IO.
    let absent = dir.join("absent.bin");
    let output = tessera(&["inspect".as_ref(), "--json".as_ref(), absent.as_os_str()]);
    assert_eq!(output.status.code(), Some(20));
    let err: Value = serde_json::from_str(text(&output.stdout)).expect("JSON error");
    assert_eq!(err["error"]["code"], "ERR-RES-IO");

    let _ = fs::remove_dir_all(&dir);
}
