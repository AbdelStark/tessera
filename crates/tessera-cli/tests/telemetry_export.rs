// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Opt-in transport behavior through the real CLI (05 §4–§5, OBS-1, #177).

use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("tessera-export-{name}-{}", std::process::id()));
    fs::create_dir_all(&path).expect("scratch directory");
    path
}

fn compile(path: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tessera"));
    command
        .args(["compile", "--model"])
        .arg(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../spec/vectors/exact-cli/tiny-3-reference.d"),
        )
        .args([
            "--params",
            "test-small",
            "--max-log2-rows",
            "18",
            "--query-block",
            "2",
            "--out-dir",
        ])
        .arg(path);
    command
}

fn success(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn telemetry_environment_alone_cannot_enable_a_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("trap listener");
    listener.set_nonblocking(true).expect("nonblocking");
    let endpoint = format!(
        "http://{}/v1/traces",
        listener.local_addr().expect("address")
    );
    let output = compile(&scratch("default"))
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", &endpoint)
        .env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", &endpoint)
        .env("OTEL_RESOURCE_ATTRIBUTES", "private=EXPORT_CANARY_177")
        .output()
        .expect("compile");
    success(&output);
    assert_eq!(
        listener
            .accept()
            .expect_err("no exporter connection")
            .kind(),
        std::io::ErrorKind::WouldBlock
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("EXPORT_CANARY_177"));
}

#[test]
fn occupied_metrics_port_fails_explicitly_before_command_outputs() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("occupied port");
    let path = scratch("occupied").join("out");
    let output = compile(&path)
        .args([
            "--metrics-addr",
            &listener.local_addr().expect("address").to_string(),
            "--json",
        ])
        .output()
        .expect("command");
    assert_eq!(output.status.code(), Some(20));
    let error: serde_json::Value = serde_json::from_slice(&output.stdout).expect("JSON failure");
    assert_eq!(error["error"]["code"], "ERR-RES-IO");
    assert!(!path.exists(), "initialization fails before compilation");
}

#[cfg(not(feature = "otel"))]
#[test]
fn default_binary_refuses_the_export_flag() {
    let output = compile(&scratch("no-feature"))
        .args(["--otlp-endpoint", "http://127.0.0.1:4318/v1/traces"])
        .output()
        .expect("command");
    assert_eq!(output.status.code(), Some(64));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--otlp-endpoint"));
}

#[cfg(feature = "otel")]
mod otel {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    fn collect(stream: &mut TcpStream) -> serde_json::Value {
        stream.set_nonblocking(false).expect("blocking");
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("timeout");
        let mut bytes = Vec::new();
        let (offset, length) = loop {
            let mut chunk = [0u8; 4096];
            let n = stream.read(&mut chunk).expect("HTTP request");
            assert_ne!(n, 0, "complete request headers");
            bytes.extend_from_slice(&chunk[..n]);
            assert!(bytes.len() < 1024 * 1024);
            if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&bytes[..end]).expect("headers");
                assert!(headers.starts_with("POST /v1/traces HTTP/1.1"));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("content-type: application/json")
                );
                let size = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().expect("content length"))
                    })
                    .expect("content length");
                break (end + 4, size);
            }
        };
        assert!(length < 1024 * 1024);
        while bytes.len() < offset + length {
            let mut chunk = [0u8; 4096];
            let n = stream.read(&mut chunk).expect("request body");
            assert_ne!(n, 0);
            bytes.extend_from_slice(&chunk[..n]);
        }
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").expect("collector response");
        serde_json::from_slice(&bytes[offset..offset + length]).expect("OTLP JSON")
    }

    #[test]
    fn explicit_export_flushes_real_spans_without_environment_resource_fields() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("collector");
        listener.set_nonblocking(true).expect("nonblocking");
        let endpoint = format!(
            "http://{}/v1/traces",
            listener.local_addr().expect("address")
        );
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let collector = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(15);
            let mut bodies = Vec::new();
            loop {
                match listener.accept() {
                    Ok((mut stream, _)) => bodies.push(collect(&mut stream)),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if stopped.load(Ordering::Relaxed) || Instant::now() >= deadline {
                            break;
                        }
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("collector: {error}"),
                }
            }
            bodies
        });
        let output = compile(&scratch("explicit"))
            .args([
                "--otlp-endpoint",
                &endpoint,
                "--log-level",
                "trace",
                "--log-format",
                "json",
            ])
            .env("OTEL_RESOURCE_ATTRIBUTES", "private=EXPORT_CANARY_177")
            .env("OTEL_SERVICE_NAME", "EXPORT_CANARY_177")
            .env("HTTP_PROXY", "http://127.0.0.1:1")
            .env("ALL_PROXY", "http://127.0.0.1:1")
            .env("NO_PROXY", "")
            .env(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                "http://127.0.0.1:1/ignored",
            )
            .output()
            .expect("compile");
        stop.store(true, Ordering::Relaxed);
        let bodies = collector.join().expect("collector thread");
        success(&output);
        for line in std::str::from_utf8(&output.stderr)
            .expect("UTF-8 events")
            .lines()
        {
            let event: serde_json::Value = serde_json::from_str(line).expect("typed JSON event");
            assert!(
                event["target"]
                    .as_str()
                    .expect("event target")
                    .starts_with("tessera_")
            );
        }
        assert!(
            !bodies.is_empty(),
            "explicit exporter sends and flushes spans"
        );
        let mut names = Vec::new();
        for body in bodies {
            let encoded = body.to_string();
            assert!(!encoded.contains("EXPORT_CANARY_177"));
            assert!(encoded.contains("tessera"));
            for resource in body["resourceSpans"].as_array().expect("resource spans") {
                for scope in resource["scopeSpans"].as_array().expect("scope spans") {
                    for span in scope["spans"].as_array().expect("spans") {
                        names.push(span["name"].as_str().expect("span name").to_owned());
                        assert!(!span["traceId"].as_str().expect("trace id").is_empty());
                        assert!(!span["spanId"].as_str().expect("span id").is_empty());
                    }
                }
            }
        }
        assert!(names.iter().any(|name| name == "compile"), "{names:?}");
    }

    #[test]
    fn collector_redirects_cannot_send_traces_to_another_destination() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("collector");
        listener.set_nonblocking(true).expect("nonblocking");
        let trap = TcpListener::bind("127.0.0.1:0").expect("redirect trap");
        trap.set_nonblocking(true).expect("nonblocking");
        let endpoint = format!(
            "http://{}/v1/traces",
            listener.local_addr().expect("address")
        );
        let destination = format!("http://{}/v1/traces", trap.local_addr().expect("address"));
        let collector = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).expect("blocking");
                        stream
                            .set_read_timeout(Some(Duration::from_secs(1)))
                            .expect("timeout");
                        let mut first = [0u8; 4096];
                        let _ = stream.read(&mut first).expect("export request");
                        write!(stream, "HTTP/1.1 307 Temporary Redirect\r\nLocation: {destination}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").expect("redirect");
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            Instant::now() < deadline,
                            "export must reach the explicit collector"
                        );
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("collector: {error}"),
                }
            }
        });
        let output = compile(&scratch("redirect"))
            .args(["--otlp-endpoint", &endpoint])
            .output()
            .expect("compile");
        collector.join().expect("collector thread");
        success(&output);
        assert_eq!(
            trap.accept()
                .expect_err("no redirected trace connection")
                .kind(),
            std::io::ErrorKind::WouldBlock
        );
    }

    #[test]
    fn collector_failure_is_bounded_and_does_not_fail_compilation() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve");
        let endpoint = format!(
            "http://{}/v1/traces",
            listener.local_addr().expect("address")
        );
        drop(listener);
        let started = Instant::now();
        let output = compile(&scratch("unreachable"))
            .args(["--otlp-endpoint", &endpoint])
            .output()
            .expect("compile");
        success(&output);
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn credentials_queries_and_non_http_endpoints_are_refused_without_echoing_values() {
        for endpoint in [
            "file:///EXPORT_CANARY_177",
            "ftp://localhost/EXPORT_CANARY_177",
            "http://EXPORT_CANARY_177@localhost/v1/traces",
            "http://:EXPORT_CANARY_177@localhost/v1/traces",
            "http://localhost/v1/traces#EXPORT_CANARY_177",
            "http://user:EXPORT_CANARY_177@localhost/v1/traces",
            "http://localhost/v1/traces?token=EXPORT_CANARY_177",
        ] {
            let output = compile(&scratch("invalid"))
                .args(["--otlp-endpoint", endpoint, "--json"])
                .output()
                .expect("command");
            assert_eq!(output.status.code(), Some(64));
            let error: serde_json::Value =
                serde_json::from_slice(&output.stdout).expect("JSON error");
            assert_eq!(error["error"]["code"], "ERR-USAGE");
            assert!(!String::from_utf8_lossy(&output.stdout).contains("EXPORT_CANARY_177"));
            assert!(!String::from_utf8_lossy(&output.stderr).contains("EXPORT_CANARY_177"));
        }
    }
}
