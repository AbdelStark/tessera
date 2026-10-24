// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera recorder ref` end to end (RFC-0015 'Reference recorder'). A client talks through the
//! recorder to an upstream. Then the test checks three things:
//! - the bytes each side received are the bytes the other sent;
//! - the logged digests match an offline recomputation;
//! - the periods written to `--log-dir` form a valid chain of manifests over their logs.

use std::fs;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use tessera_anchor::recorder::{WireFrame, decode_log};
use tessera_anchor::{
    DigestLogEntry, Direction, HeapKey, LogManifest, RequestEndFrame, Session, TokenFrame,
    check_chain, merkle, public_key,
};
use tessera_codec::{DecodeLimits, Secret, envelope, schema};

const SEED: [u8; 32] = [0x2a; 32];

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-recorder-{test}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn token(request: u64, direction: Direction, index: u32, tokens: Vec<u32>) -> TokenFrame {
    TokenFrame::new(request, direction, index, Secret::new(tokens))
}

/// The `.tlm` and `.tdl` files of every period written so far, in `seq` order.
fn periods(dir: &Path) -> Vec<(LogManifest, Vec<DigestLogEntry>)> {
    let mut stems: Vec<String> = fs::read_dir(dir)
        .expect("the log directory")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            name.strip_suffix(".tlm").map(str::to_owned)
        })
        .collect();
    stems.sort();
    stems
        .iter()
        .map(|stem| {
            let tlm = fs::read(dir.join(format!("{stem}.tlm"))).expect("a manifest");
            let manifest: LogManifest =
                envelope::open(&tlm, &schema::LOG_MANIFEST, DecodeLimits::MESSAGE)
                    .expect("a manifest envelope");
            let log = fs::read(dir.join(format!("{stem}.tdl"))).expect("a period's log");
            (manifest, decode_log(&log).expect("a period's entries"))
        })
        .collect()
}

#[test]
fn the_recorder_forwards_logs_and_chains_its_periods() {
    let dir = scratch("e2e");
    let log_dir = dir.join("log");
    let key = dir.join("recorder.seed");
    fs::write(&key, SEED).expect("the seed");

    // The frames: one session, request 3's prompt in, its answer out and its end.
    let (mut session, open, _) = Session::open().expect("a session");
    let id = session.session_id();
    let inbound = [
        WireFrame::Open(open),
        WireFrame::Sealed(
            session
                .seal(&token(3, Direction::In, 0, vec![10, 20, 30]))
                .expect("sealed"),
        ),
    ];
    let outbound = [
        WireFrame::Sealed(
            session
                .seal(&token(3, Direction::Out, 0, vec![40; 12]))
                .expect("sealed"),
        ),
        WireFrame::End(RequestEndFrame {
            session_id: id,
            request_id: 3,
        }),
    ];
    let inbound_bytes: Vec<u8> = inbound.iter().flat_map(WireFrame::to_bytes).collect();
    let outbound_bytes: Vec<u8> = outbound.iter().flat_map(WireFrame::to_bytes).collect();

    // The upstream: reads the inbound direction to its end, then answers.
    let upstream = TcpListener::bind("127.0.0.1:0").expect("an upstream port");
    let upstream_addr = upstream.local_addr().expect("its address");
    let answer = outbound_bytes.clone();
    let server = thread::spawn(move || {
        let (mut conn, _) = upstream.accept().expect("the recorder connects");
        let mut received = Vec::new();
        conn.read_to_end(&mut received).expect("the inbound bytes");
        conn.write_all(&answer).expect("the answer");
        conn.shutdown(Shutdown::Write)
            .expect("the end of the answer");
        received
    });

    let mut child = Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(["recorder", "ref", "--listen", "127.0.0.1:0", "--upstream"])
        .arg(upstream_addr.to_string())
        .arg("--log-dir")
        .arg(&log_dir)
        .arg("--key")
        .arg(&key)
        .args(["--period", "1"])
        .stderr(Stdio::piped())
        .spawn()
        .expect("the recorder starts");
    let mut stderr = child.stderr.take().expect("its stderr");
    // Its first line, read a byte at a time: no read-ahead buffer (OBS-6).
    let mut line = Vec::new();
    let mut byte = [0u8; 1];
    while stderr.read(&mut byte).expect("its stderr") == 1 && byte[0] != b'\n' {
        line.push(byte[0]);
    }
    let line = String::from_utf8(line).expect("a UTF-8 line");
    let address = line
        .split("listening on ")
        .nth(1)
        .and_then(|rest| rest.split(',').next())
        .expect("the listening address")
        .trim()
        .to_owned();
    // Keep reading stderr, so the recorder never blocks on a full pipe.
    thread::spawn(move || {
        let mut rest = String::new();
        let _ = stderr.read_to_string(&mut rest);
    });

    let mut client = TcpStream::connect(&address).expect("the client connects");
    client.write_all(&inbound_bytes).expect("the inbound bytes");
    client
        .shutdown(Shutdown::Write)
        .expect("the end of the prompt");
    let mut received = Vec::new();
    client.read_to_end(&mut received).expect("the answer");
    drop(client);
    let upstream_received = server.join().expect("the upstream");

    // Bytes forwarded unchanged, in both directions.
    assert_eq!(upstream_received, inbound_bytes);
    assert_eq!(received, outbound_bytes);

    // Wait for two periods to close after the conversation.
    let deadline = Instant::now() + Duration::from_secs(20);
    let written = loop {
        let written = periods(&log_dir);
        let logged: usize = written.iter().map(|(_, e)| e.len()).sum();
        if written.len() >= 2 && logged >= 5 {
            break written;
        }
        assert!(Instant::now() < deadline, "no periods written");
        thread::sleep(Duration::from_millis(100));
    };
    child.kill().expect("the recorder stops");
    let _ = child.wait();

    // A valid chain from genesis under the recorder's key, each root over its log.
    let seed = Secret::new(HeapKey::from_box(Box::new(SEED)));
    let manifests: Vec<LogManifest> = written.iter().map(|(m, _)| m.clone()).collect();
    assert_eq!(check_chain(&manifests, &[public_key(&seed)]), Ok(()));
    let mut entries = Vec::new();
    for (manifest, log) in &written {
        let leaves: Vec<_> = log.iter().map(merkle::leaf_hash).collect();
        assert_eq!(manifest.body.entries_root, merkle::root(&leaves));
        assert_eq!(manifest.body.entry_count, log.len() as u64);
        entries.extend(log.iter().copied());
    }

    // The digests match an offline recomputation, and the session's close comes last.
    let sealed: Vec<_> = inbound
        .iter()
        .chain(&outbound)
        .filter_map(|f| match f {
            WireFrame::Sealed(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let logged_frames: Vec<_> = entries
        .iter()
        .filter(|e| matches!(e, DigestLogEntry::Frame { .. }))
        .collect();
    assert_eq!(logged_frames.len(), sealed.len());
    for (entry, frame) in logged_frames.iter().zip(&sealed) {
        assert_eq!(
            **entry,
            DigestLogEntry::frame(entry.seq(), entry.t_ns(), frame)
        );
    }
    let kinds: Vec<&str> = entries
        .iter()
        .filter(|e| !matches!(e, DigestLogEntry::SessionCarry { .. }))
        .map(|e| match e {
            DigestLogEntry::SessionOpen { .. } => "open",
            DigestLogEntry::Frame { .. } => "frame",
            DigestLogEntry::RequestEnd { .. } => "end",
            DigestLogEntry::SessionClose { .. } => "close",
            DigestLogEntry::SessionCarry { .. } => "carry",
        })
        .collect();
    assert_eq!(kinds, ["open", "frame", "frame", "end", "close"]);
    // Inbound frames were logged before the outbound ones (RFC-0063 rule 3).
    let first_out = entries
        .iter()
        .position(|e| {
            matches!(
                e,
                DigestLogEntry::Frame {
                    direction: Direction::Out,
                    ..
                }
            )
        })
        .expect("an outbound frame");
    assert!(entries[..first_out].iter().any(|e| matches!(
        e,
        DigestLogEntry::Frame {
            direction: Direction::In,
            ..
        }
    )));
    let _ = fs::remove_dir_all(&dir);
}
