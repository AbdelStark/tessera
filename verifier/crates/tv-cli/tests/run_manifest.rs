// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera-verify verify --run-manifest PATH` (05 §6 'The verifier's manifest',
//! `spec/schemas/run-manifest-v1.schema.json`): public-class, written after the report and only
//! at `PATH`, which must differ from every input and from `--out`; a manifest that cannot be
//! written fails the command.

mod observe_support;
use observe_support::{
    Field, Fixture, accepts_test_small, assert_timestamp, envelope_digest, lines,
};
use std::fs;
use std::path::Path;
use tv_cli::run;
use tv_testkit::Json;

fn object_keys(value: &Json) -> Vec<&str> {
    let Json::Object(members) = value else {
        panic!("{value:?} is not an object");
    };
    let mut keys: Vec<&str> = members.iter().map(|(k, _)| k.as_str()).collect();
    keys.sort_unstable();
    keys
}
fn get<'a>(value: &'a Json, key: &str) -> &'a Json {
    value.get(key).unwrap_or_else(|| panic!("no {key}"))
}
fn text<'a>(value: &'a Json, key: &str) -> &'a str {
    get(value, key)
        .as_str()
        .unwrap_or_else(|| panic!("{key} is not text"))
}
fn read(path: &Path) -> Json {
    Json::parse(&fs::read_to_string(path).unwrap()).unwrap()
}
/// A JSON number written as decimal seconds, `S.fffffffff`, in nanoseconds.
fn nanos(value: &Json) -> u64 {
    let Json::Number(n) = value else {
        panic!("{value:?} is not a number");
    };
    let (s, f) = n.split_once('.').unwrap_or((n, ""));
    assert!(f.len() <= 9, "{n}");
    let f = format!("{f:0<9}");
    s.parse::<u64>().unwrap() * 1_000_000_000 + f.parse::<u64>().unwrap()
}
fn artifact(value: &Json) -> (&str, &str, &str, Option<&str>) {
    assert_eq!(object_keys(value), ["class", "digest", "path", "schema"]);
    (
        text(value, "path"),
        text(value, "schema"),
        text(value, "class"),
        get(value, "digest").as_str(),
    )
}
const KEYS: [&str; 16] = [
    "arguments",
    "command",
    "elapsed_seconds",
    "error_code",
    "exit_code",
    "host",
    "inputs",
    "metrics",
    "outputs",
    "params",
    "protocol",
    "public_seeds",
    "schema",
    "started_at",
    "tool",
    "toolchain",
];

#[test]
fn the_manifest_names_the_build_the_inputs_and_each_check() {
    let f = Fixture::new();
    let out = f.dir.join("verdict.json");
    let manifest = f.dir.join("run-manifest.json");
    let args = f.with(&[
        "--out",
        &out.display().to_string(),
        "--run-manifest",
        &manifest.display().to_string(),
        "--log-format",
        "json",
        "--log-level",
        "debug",
    ]);
    let result = run(&args);
    let m = read(&manifest);
    assert_eq!(object_keys(&m), KEYS);
    assert_eq!(text(&m, "schema"), "tessera.run-manifest/1");
    let tool = get(&m, "tool");
    assert_eq!(
        object_keys(tool),
        ["features", "git_commit", "name", "profile", "version"]
    );
    assert_eq!(text(tool, "name"), "tessera-verify");
    assert_eq!(text(tool, "version"), env!("CARGO_PKG_VERSION"));
    // Without a build script (check-isolation), the commit is the one the build was told.
    let told = option_env!("TESSERA_GIT_COMMIT")
        .filter(|c| c.len() == 40 && c.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(str::to_ascii_lowercase);
    assert_eq!(get(tool, "git_commit").as_str(), told.as_deref());
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    assert_eq!(text(tool, "profile"), profile);
    assert_eq!(get(tool, "features"), &expected_features());
    let toolchain = get(&m, "toolchain");
    assert_eq!(
        object_keys(toolchain),
        ["cuda_driver", "cuda_runtime", "python", "rustc", "torch"]
    );
    // The compiler's version is the one the build was told, as the commit is.
    let rustc = option_env!("TESSERA_RUSTC_VERSION").filter(|v| !v.is_empty());
    assert_eq!(get(toolchain, "rustc").as_str(), rustc);
    for key in ["cuda_driver", "cuda_runtime", "python", "torch"] {
        assert!(get(toolchain, key).is_null(), "{key}");
    }
    let host = get(&m, "host");
    assert_eq!(
        object_keys(host),
        [
            "arch",
            "cores",
            "cpu_model",
            "ecc",
            "gpus",
            "memory_bytes",
            "os"
        ]
    );
    assert_eq!(text(host, "os"), std::env::consts::OS);
    assert_eq!(text(host, "arch"), std::env::consts::ARCH);
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .ok();
    assert_eq!(get(host, "cores").as_u64(), cores);
    assert!(get(host, "cpu_model").is_null() || get(host, "cpu_model").as_str().is_some());
    let memory = get(host, "memory_bytes");
    assert!(memory.is_null() || memory.as_u64().is_some_and(|n| n > 0));
    if cfg!(any(target_os = "linux", target_os = "macos")) {
        assert!(!memory.is_null(), "the platform reports its memory");
    }
    assert_eq!(get(host, "gpus"), &Json::Array(vec![]));
    assert!(get(host, "ecc").is_null());
    assert_eq!(text(&m, "command"), "verify");
    let arguments: Vec<&str> = get(&m, "arguments")
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect();
    assert_eq!(arguments, args);
    assert_eq!(text(&m, "params"), "test-small/1");
    assert_eq!(text(&m, "protocol"), "tessera-protocol/1");
    let inputs: Vec<_> = get(&m, "inputs")
        .as_array()
        .unwrap()
        .iter()
        .map(artifact)
        .collect();
    let claims = envelope_digest(&f.claims);
    let weights = envelope_digest(&f.weights);
    let proof = envelope_digest(&f.proof);
    assert_eq!(
        inputs,
        [
            (
                f.claims.to_str().unwrap(),
                "ClaimList",
                "public",
                Some(claims.as_str())
            ),
            (
                f.weights.to_str().unwrap(),
                "WeightCommitmentPublic",
                "public",
                Some(weights.as_str())
            ),
            (
                f.proof.to_str().unwrap(),
                "NiProof",
                "protocol",
                Some(proof.as_str())
            ),
        ]
    );
    let outputs: Vec<_> = get(&m, "outputs")
        .as_array()
        .unwrap()
        .iter()
        .map(artifact)
        .collect();
    assert_eq!(
        outputs,
        [(out.to_str().unwrap(), "tessera.verdict/1", "public", None)]
    );
    assert_eq!(get(&m, "public_seeds"), &Json::Array(vec![]));
    assert_timestamp(text(&m, "started_at"));
    nanos(get(&m, "elapsed_seconds"));
    // One histogram sample per check that ran, its time the debug line's.
    let checks: Vec<(String, u64)> = lines(&result.stderr)
        .iter()
        .filter(|l| l.event == "tessera.tv_cli.check_completed")
        .map(|l| {
            let kind = l.spans[1].fields[0].1.text().to_string();
            (kind, l.field("duration_ns").map(Field::int).unwrap())
        })
        .collect();
    let metrics = get(&m, "metrics").as_array().unwrap();
    assert_eq!(metrics.len(), checks.len());
    assert_eq!(checks.len(), if accepts_test_small() { 13 } else { 2 });
    for (sample, (kind, ns)) in metrics.iter().zip(&checks) {
        assert_eq!(
            object_keys(sample),
            ["count", "labels", "max", "min", "name", "sum", "type"]
        );
        assert_eq!(text(sample, "name"), "tessera_verifier_seconds");
        assert_eq!(text(sample, "type"), "histogram");
        assert_eq!(object_keys(get(sample, "labels")), ["check"]);
        assert_eq!(text(get(sample, "labels"), "check"), kind);
        assert_eq!(get(sample, "count").as_u64(), Some(1));
        for key in ["sum", "min", "max"] {
            assert_eq!(nanos(get(sample, key)), *ns, "{kind} {key}");
        }
    }
    assert_eq!(
        get(&m, "exit_code").as_u64(),
        u64::try_from(result.exit_code).ok()
    );
    if accepts_test_small() {
        assert_eq!(result.exit_code, 0);
        assert!(get(&m, "error_code").is_null());
    } else {
        assert_eq!(result.exit_code, 10);
        assert_eq!(text(&m, "error_code"), "REJ-PARAMS");
    }
    // Written atomically: nothing but the inputs, the report and the manifest remains.
    assert_eq!(fs::read_dir(&f.dir).unwrap().count(), 5);
}

/// The test features this build enables: `tv-core/test-params` exactly when it accepts test-small.
fn expected_features() -> Json {
    if accepts_test_small() {
        Json::Array(vec![Json::String("tv-core/test-params".into())])
    } else {
        Json::Array(vec![])
    }
}

/// A plain build lists no feature; one built with `--features tv-core/test-params`, which
/// certifies nothing, says so in every manifest, whatever the verdict.
#[test]
fn the_manifest_lists_the_test_features_of_the_build() {
    let f = Fixture::new();
    for (proof, exit) in [(None, None), (Some(&b"not an envelope"[..]), Some(65))] {
        if let Some(bytes) = proof {
            fs::write(&f.proof, bytes).unwrap();
        }
        let manifest = f.dir.join("run-manifest.json");
        let result = run(f.with(&["--run-manifest", &manifest.display().to_string()]));
        if let Some(exit) = exit {
            assert_eq!(result.exit_code, exit);
        }
        let m = read(&manifest);
        let features = get(get(&m, "tool"), "features");
        assert_eq!(features, &expected_features());
        if accepts_test_small() {
            assert_eq!(text(&m, "params"), "test-small/1");
        }
    }
}

#[test]
fn a_manifest_path_that_names_an_input_or_the_report_is_usage_before_verification() {
    let f = Fixture::new();
    let out = f.dir.join("verdict.json");
    for target in [&f.claims, &f.weights, &f.proof, &out] {
        let before: Vec<Vec<u8>> = [&f.claims, &f.weights, &f.proof]
            .iter()
            .map(|p| fs::read(p).unwrap())
            .collect();
        let result = run(f.with(&[
            "--out",
            &out.display().to_string(),
            "--run-manifest",
            &target.display().to_string(),
            "--json",
        ]));
        assert_eq!(result.exit_code, 64, "{}", target.display());
        let verdict = Json::parse(&result.stdout).unwrap();
        let error = get(get(get(&verdict, "verdict"), "error"), "code");
        assert_eq!(error.as_str(), Some("ERR-USAGE"));
        let after: Vec<Vec<u8>> = [&f.claims, &f.weights, &f.proof]
            .iter()
            .map(|p| fs::read(p).unwrap())
            .collect();
        assert_eq!(before, after);
        assert!(!out.exists(), "nothing is written");
        assert_eq!(fs::read_dir(&f.dir).unwrap().count(), 3);
    }
    // The same path spelled through its directory is still the same file.
    let dotted = f.dir.join(".").join("verdict.json");
    let result = run(f.with(&[
        "--out",
        &out.display().to_string(),
        "--run-manifest",
        &dotted.display().to_string(),
    ]));
    assert_eq!(result.exit_code, 64);
    assert!(!out.exists());
}

#[test]
fn a_manifest_that_cannot_be_written_fails_the_command_and_the_report_stays() {
    let f = Fixture::new();
    let out = f.dir.join("verdict.json");
    let blocked = f.dir.join("blocked");
    fs::create_dir(&blocked).unwrap();
    let result = run(f.with(&[
        "--out",
        &out.display().to_string(),
        "--run-manifest",
        &blocked.display().to_string(),
        "--json",
        "--log-format",
        "json",
    ]));
    assert_eq!(result.exit_code, 20);
    let stdout = Json::parse(&result.stdout).unwrap();
    let verdict = get(&stdout, "verdict");
    assert_eq!(text(verdict, "status"), "inconclusive");
    assert_eq!(text(get(verdict, "error"), "code"), "ERR-RES-IO");
    // The report written before the manifest keeps the verification's verdict.
    let report = read(&out);
    let status = text(get(&report, "verdict"), "status");
    assert_eq!(
        status,
        if accepts_test_small() {
            "accept"
        } else {
            "reject"
        }
    );
    assert!(blocked.is_dir() && fs::read_dir(&blocked).unwrap().count() == 0);
    assert_eq!(fs::read_dir(&f.dir).unwrap().count(), 5);
    // `verified` reports the command's final status.
    let last = lines(&result.stderr).pop().unwrap();
    assert_eq!(last.event, "tessera.tv_cli.verified");
    assert_eq!(last.field("status").unwrap().text(), "inconclusive");
    assert_eq!(last.field("code").unwrap().text(), "ERR-RES-IO");
    assert_eq!(last.field("exit_code").unwrap().int(), 20);
    // A directory that does not exist cannot hold one either.
    let missing = f.dir.join("missing").join("run-manifest.json");
    let result = run(f.with(&["--run-manifest", &missing.display().to_string()]));
    assert_eq!(result.exit_code, 20);
}

#[test]
fn the_manifest_is_written_after_the_report_and_records_the_final_exit() {
    let f = Fixture::new();
    let blocked = f.dir.join("blocked");
    fs::create_dir(&blocked).unwrap();
    let manifest = f.dir.join("run-manifest.json");
    let result = run(f.with(&[
        "--out",
        &blocked.display().to_string(),
        "--run-manifest",
        &manifest.display().to_string(),
    ]));
    assert_eq!(result.exit_code, 20);
    let m = read(&manifest);
    assert_eq!(get(&m, "exit_code").as_u64(), Some(20));
    assert_eq!(text(&m, "error_code"), "ERR-RES-IO");
    assert_eq!(get(&m, "outputs"), &Json::Array(vec![]));
}

/// An input replaced, and what the manifest then records.
struct Broken {
    file: &'static str,
    /// The replacement; empty for the proof's envelope resealed over a one-byte payload.
    bytes: &'static [u8],
    error_code: &'static str,
    /// Which of the claim list, weights and proof have a digest.
    digests: [bool; 3],
    params: bool,
}

#[test]
fn an_artifact_that_fails_to_open_is_named_without_a_digest() {
    for case in [
        Broken {
            file: "ni.tnp",
            bytes: b"not an envelope",
            error_code: "ERR-CODEC-MAGIC",
            digests: [true, true, false],
            params: true,
        },
        Broken {
            file: "claims.tcl",
            bytes: b"not an envelope",
            error_code: "ERR-CODEC-MAGIC",
            digests: [false, false, false],
            params: false,
        },
        // A sound envelope whose payload fails: its digest is known once the envelope verifies.
        Broken {
            file: "ni.tnp",
            bytes: &[],
            error_code: "ERR-CODEC-TRUNCATED",
            digests: [true, true, true],
            params: true,
        },
    ] {
        let f = Fixture::new();
        let target = f.dir.join(case.file);
        let replacement = if case.bytes.is_empty() {
            let mut out = fs::read(&target).unwrap()[..12].to_vec();
            out.extend_from_slice(&1u64.to_le_bytes());
            out.push(0);
            let digest = tv_hash::hash(tv_hash::Domain::Envelope, &out);
            out.extend_from_slice(&digest);
            out
        } else {
            case.bytes.to_vec()
        };
        fs::write(&target, &replacement).unwrap();
        let manifest = f.dir.join("run-manifest.json");
        let result = run(f.with(&["--run-manifest", &manifest.display().to_string()]));
        assert_eq!(result.exit_code, 65, "{}", target.display());
        let m = read(&manifest);
        let digests: Vec<bool> = get(&m, "inputs")
            .as_array()
            .unwrap()
            .iter()
            .map(|i| !get(i, "digest").is_null())
            .collect();
        assert_eq!(digests, case.digests, "{}", target.display());
        if case.digests[2] {
            let proof = get(&m, "inputs").as_array().unwrap()[2].clone();
            assert_eq!(
                get(&proof, "digest").as_str(),
                Some(envelope_digest(&target).as_str())
            );
        }
        assert_eq!(get(&m, "params").is_null(), !case.params);
        assert_eq!(text(&m, "error_code"), case.error_code);
        assert_eq!(get(&m, "exit_code").as_u64(), Some(65));
        let metrics = get(&m, "metrics").as_array().unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(text(get(&metrics[0], "labels"), "check"), "input");
    }
}
