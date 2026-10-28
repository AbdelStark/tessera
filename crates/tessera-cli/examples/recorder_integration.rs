// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A synthetic serving adapter through the real reference recorder (#180).
//! Run: `cargo run -p tessera-cli --example recorder_integration -- TESSERA_BIN NEW_DIR`.
//! See docs/integration.md. This uses a PUBLIC test recorder key, a toy distribution,
//! and a simulated sampler issuer; it establishes no production placement or proof claim.

#![forbid(unsafe_code)]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tessera_anchor::recorder::decode_log;
use tessera_anchor::{
    BatchRequest, ChainError, DigestLogEntry, Direction, HeapKey, LogManifest, LogMismatch,
    ManifestError, PrecheckKind, RequestEndFrame, SealedFrame, Session, SessionTranscript,
    TokenFrame, VerifiedLog, check_chain, precheck, public_key, sampler_commitment,
};
use tessera_codec::{DecodeLimits, Secret, SecretBuf, envelope, schema};
use tessera_scoring::{ClassBits, Sampler, SamplerBaseKey, inverse_cdf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
type Period = (LogManifest, Vec<DigestLogEntry>);

const PUBLIC_TEST_SEED: [u8; 32] = [0x2a; 32];
const REQUEST: u64 = 7;
const IO_TIMEOUT: Duration = Duration::from_secs(5);

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn seal_record(
    session: &mut Session,
    transcript: &mut SessionTranscript,
    frame: TokenFrame,
) -> Result<SealedFrame> {
    let sealed = session.seal(&frame)?;
    // Move the plaintext into clearing storage before sending the ciphertext.
    transcript.record(&sealed, frame)?;
    Ok(sealed)
}

struct Exchange {
    transcript: SessionTranscript,
    frames: Vec<SealedFrame>,
    inbound: Vec<u8>,
    outbound: Vec<u8>,
}

fn prepared() -> Result<Exchange> {
    let (mut session, open, keys) = Session::open()?;
    let mut transcript = SessionTranscript::new(session.session_id(), keys);
    let prompt = Secret::new(vec![1, 2, 3]);

    let base = SamplerBaseKey::new(HeapKey::generate()?);
    let salt = HeapKey::generate()?;
    let commitment = sampler_commitment(base.secret(), &salt);
    // Simulated issuer only: production sends this commitment over the
    // authenticated pin channel BEFORE the verifier draws and returns its nonce.
    let mut nonce = [0u8; 32];
    getrandom::fill(&mut nonce)?;
    let from_ns = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let bits = ClassBits::new(0).ok_or("invalid class bits")?;
    let key = base.derive(&nonce, bits, from_ns);
    let sampler = Sampler::new(&key);
    let class = sampler.class(&prompt);
    // Commitment is public, but neither it nor a local random nonce constitutes
    // a verified pin. The integration guide describes the missing issuer policy.
    let _ = commitment;

    let mut frames = vec![seal_record(
        &mut session,
        &mut transcript,
        TokenFrame::new(REQUEST, Direction::In, 0, prompt),
    )?];
    for position in 0..2u32 {
        // Synthetic public weights, NOT an LLM's logits. The draw and sampled
        // token remain secret-class scalars and never enter output or errors.
        let draw = sampler.draw(class, position);
        let token = u32::try_from(inverse_cdf(&[0, 1, 3, 0], draw)?)?;
        frames.push(seal_record(
            &mut session,
            &mut transcript,
            TokenFrame::new(REQUEST, Direction::Out, position, Secret::new(vec![token])),
        )?);
    }
    let mut inbound = open.to_bytes().to_vec();
    inbound.extend(frames[0].to_bytes());
    let mut outbound: Vec<u8> = frames[1..].iter().flat_map(SealedFrame::to_bytes).collect();
    outbound.extend(
        RequestEndFrame {
            session_id: session.session_id(),
            request_id: REQUEST,
        }
        .to_bytes(),
    );
    Ok(Exchange {
        transcript,
        frames,
        inbound,
        outbound,
    })
}

fn load_transcript(path: &Path) -> Result<SessionTranscript> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    require(
        size <= DecodeLimits::MESSAGE.max_total_bytes,
        "transcript exceeds its byte limit",
    )?;
    let size = usize::try_from(size)?;
    let mut bytes = SecretBuf::try_with_capacity(size)?;
    bytes.resize(size, 0u8);
    file.read_exact(&mut bytes)?;
    let mut extra = SecretBuf::filled(0u8, 1);
    require(
        file.read(&mut extra)? == 0,
        "transcript changed while reading",
    )?;
    Ok(envelope::open(
        &bytes,
        &schema::SESSION_TRANSCRIPT,
        DecodeLimits::MESSAGE,
    )?)
}

/// Always stop the reference recorder and join its stderr drainer, on errors too.
struct Recorder {
    child: Child,
    stderr: Option<JoinHandle<()>>,
}

impl Drop for Recorder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(stderr) = self.stderr.take() {
            let _ = stderr.join();
        }
    }
}

fn start(binary: &Path, directory: &Path, upstream: SocketAddr) -> Result<(Recorder, SocketAddr)> {
    let child = Command::new(binary)
        .args(["recorder", "ref", "--listen", "127.0.0.1:0", "--upstream"])
        .arg(upstream.to_string())
        .arg("--log-dir")
        .arg(directory.join("logs"))
        .arg("--key")
        .arg(directory.join("PUBLIC-TEST-ONLY.seed"))
        .args(["--period", "1"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut recorder = Recorder {
        child,
        stderr: None,
    };
    let mut stderr = recorder
        .child
        .stderr
        .take()
        .ok_or("missing recorder stderr")?;
    let (ready, received) = mpsc::channel();
    recorder.stderr = Some(thread::spawn(move || {
        // No read-ahead buffer: keep even diagnostic bytes in clearing storage.
        let mut line = SecretBuf::with_capacity(1024);
        let mut byte = SecretBuf::filled(0u8, 1);
        while line.len() < 1024 && stderr.read(&mut byte).ok() == Some(1) {
            if byte[0] == b'\n' {
                break;
            }
            line.push(byte[0]);
        }
        let address = std::str::from_utf8(&line).ok().and_then(|line| {
            line.split("listening on ")
                .nth(1)?
                .split(',')
                .next()?
                .trim()
                .parse::<SocketAddr>()
                .ok()
        });
        let _ = ready.send(address);
        let mut tail = SecretBuf::filled(0u8, 4096);
        while stderr.read(&mut tail).is_ok_and(|n| n != 0) {}
    }));
    let address = received
        .recv_timeout(IO_TIMEOUT)?
        .ok_or("recorder did not become ready")?;
    Ok((recorder, address))
}

fn periods(directory: &Path) -> Result<Vec<Period>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(directory)?
        .map(|e| e.map(|v| v.path()))
        .collect::<std::result::Result<_, _>>()?;
    paths.retain(|p| p.extension().is_some_and(|e| e == "tlm"));
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let manifest = envelope::open(
                &fs::read(&path)?,
                &schema::LOG_MANIFEST,
                DecodeLimits::MESSAGE,
            )?;
            let entries = decode_log(&fs::read(path.with_extension("tdl"))?)?;
            Ok((manifest, entries))
        })
        .collect()
}

fn authenticate(periods: &[Period], pinned: [u8; 32]) -> Result<Vec<VerifiedLog>> {
    let manifests: Vec<_> = periods.iter().map(|(m, _)| m.clone()).collect();
    // Authenticate the complete chain FIRST. A key bundled with a signature is
    // not its own trust anchor (RFC-0060, RFC-0062).
    check_chain(&manifests, &[pinned])?;
    periods
        .iter()
        .map(|(manifest, entries)| Ok(VerifiedLog::check(manifest, entries.clone())?))
        .collect()
}

fn check_recording(
    recorded: &[Period],
    frames: &[SealedFrame],
    transcript: SessionTranscript,
) -> Result<()> {
    let seed = Secret::new(HeapKey::from_box(Box::new(PUBLIC_TEST_SEED)));
    let pinned = public_key(&seed);
    let verified = authenticate(recorded, pinned)?;
    let entries: Vec<_> = verified
        .iter()
        .flat_map(VerifiedLog::entries)
        .copied()
        .collect();
    let logged: Vec<_> = entries
        .iter()
        .filter(|e| matches!(e, DigestLogEntry::Frame { .. }))
        .collect();
    require(logged.len() == frames.len(), "wrong recorded frame count")?;
    for (entry, frame) in logged.iter().zip(frames) {
        require(
            **entry == DigestLogEntry::frame(entry.seq(), entry.t_ns(), frame),
            "recorded frame differs from wire frame",
        )?;
    }
    let end_period = verified
        .iter()
        .position(|p| {
            p.entries().iter().any(|e| {
                matches!(
                    e,
                    DigestLogEntry::RequestEnd {
                        request_id: REQUEST,
                        ..
                    }
                )
            })
        })
        .ok_or("request end is missing")?;
    let current = &verified[end_period];
    let previous = end_period.checked_sub(1).map(|i| &verified[i]);
    let request = BatchRequest {
        session_id: transcript.session_id(),
        request_id: REQUEST,
        prompt_len: 3,
        output_len: 2,
    };
    let transcripts = [transcript];
    precheck(&[request], &transcripts, current, previous)?;

    // Negative controls exercise the same helpers used for the successful path.
    let wrong_key = authenticate(recorded, [0; 32]).expect_err("an unpinned key must fail");
    require(
        wrong_key.downcast_ref::<ChainError>()
            == Some(&ChainError::Manifest {
                index: 0,
                error: ManifestError::Unpinned,
            }),
        "wrong key failed for the wrong reason",
    )?;
    let mut tampered = recorded.to_vec();
    let frame = tampered
        .iter_mut()
        .flat_map(|(_, entries)| entries)
        .find(|e| matches!(e, DigestLogEntry::Frame { .. }))
        .ok_or("missing frame for negative control")?;
    if let DigestLogEntry::Frame { sha256, .. } = frame {
        sha256[0] ^= 1;
    }
    let changed = authenticate(&tampered, pinned).expect_err("changed log must fail");
    require(
        changed.downcast_ref::<LogMismatch>() == Some(&LogMismatch::Root),
        "changed log failed for the wrong reason",
    )?;
    let wrong_length = BatchRequest {
        output_len: 3,
        ..request
    };
    let failure = precheck(&[wrong_length], &transcripts, current, previous)
        .expect_err("wrong length must fail");
    require(
        failure.kind == PrecheckKind::Lengths,
        "wrong length failed for the wrong reason",
    )?;
    Ok(())
}

fn run(binary: &Path, directory: &Path) -> Result<()> {
    // Refuse existing paths: reruns must never overwrite a transcript or recorder key.
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(directory)?;
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(directory.join("PUBLIC-TEST-ONLY.seed"))?
        .write_all(&PUBLIC_TEST_SEED)?;
    let Exchange {
        transcript,
        frames,
        inbound,
        outbound,
    } = prepared()?;
    envelope::write_file(
        &directory.join("session.tst"),
        &schema::SESSION_TRANSCRIPT,
        &transcript,
    )?;
    // This synthetic peer already knows the toy exchange. It demonstrates the
    // wire adapter, not decryption, D inference, TLS, or production key sharing.
    let upstream = TcpListener::bind("127.0.0.1:0")?;
    upstream.set_nonblocking(true)?;
    let address = upstream.local_addr()?;
    let expected_in = inbound.clone();
    let expected_out = outbound.clone();
    let server = thread::spawn(move || -> Result<()> {
        let deadline = Instant::now() + IO_TIMEOUT;
        let mut conn = loop {
            match upstream.accept() {
                Ok((conn, _)) => break conn,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(10))
                }
                Err(e) => return Err(e.into()),
            }
        };
        conn.set_nonblocking(false)?;
        conn.set_read_timeout(Some(IO_TIMEOUT))?;
        conn.set_write_timeout(Some(IO_TIMEOUT))?;
        let mut received = vec![0u8; expected_in.len()];
        conn.read_exact(&mut received)?;
        require(received == expected_in, "inbound forwarding changed bytes")?;
        conn.write_all(&expected_out)?;
        conn.shutdown(Shutdown::Write)?;
        Ok(())
    });
    let (recorder, address) = start(binary, directory, address)?;
    let mut client = TcpStream::connect_timeout(&address, IO_TIMEOUT)?;
    client.set_read_timeout(Some(IO_TIMEOUT))?;
    client.set_write_timeout(Some(IO_TIMEOUT))?;
    // Fragment the stream across magic/header boundaries; TCP packet boundaries
    // do not define frames (RFC-0015 'Reference recorder').
    for chunk in inbound.chunks(7) {
        client.write_all(chunk)?;
    }
    client.shutdown(Shutdown::Write)?;
    let mut received = vec![0u8; outbound.len()];
    client.read_exact(&mut received)?;
    require(received == outbound, "outbound forwarding changed bytes")?;
    require(
        client.read(&mut [0u8; 1])? == 0,
        "unexpected trailing bytes",
    )?;
    drop(client);
    server.join().map_err(|_| "synthetic peer failed")??;
    let deadline = Instant::now() + Duration::from_secs(20);
    let recorded = loop {
        let recorded = periods(&directory.join("logs"))?;
        let closed = recorded
            .iter()
            .flat_map(|(_, e)| e)
            .any(|e| matches!(e, DigestLogEntry::SessionClose { .. }));
        if closed
            && recorded
                .last()
                .is_some_and(|(_, entries)| entries.is_empty())
        {
            break recorded;
        }
        require(
            Instant::now() < deadline,
            "recorder period deadline exceeded",
        )?;
        thread::sleep(Duration::from_millis(50));
    };
    drop(recorder);
    // Read back the private artifact instead of trusting only the in-memory copy.
    let saved = load_transcript(&directory.join("session.tst"))?;
    check_recording(&recorded, &frames, saved)?;
    // A signed, internally consistent log must still match the bytes this
    // adapter actually sent; testing its root alone would miss this check.
    let mut changed_wire = frames.clone();
    changed_wire[0].ciphertext[0] ^= 1;
    let changed = check_recording(
        &recorded,
        &changed_wire,
        load_transcript(&directory.join("session.tst"))?,
    )
    .expect_err("wire mismatch must fail");
    require(
        changed.to_string() == "recorded frame differs from wire frame",
        "wire mismatch failed for the wrong reason",
    )?;
    let missing = check_recording(
        &recorded,
        &frames[..2],
        load_transcript(&directory.join("session.tst"))?,
    )
    .expect_err("missing wire frame must fail");
    require(
        missing.to_string() == "wrong recorded frame count",
        "missing frame failed for the wrong reason",
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        require(
            fs::metadata(directory.join("session.tst"))?
                .permissions()
                .mode()
                & 0o777
                == 0o600,
            "transcript permissions are not 0600",
        )?;
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: recorder_integration TESSERA_BIN NEW_DIR");
        return ExitCode::from(64);
    }
    match run(Path::new(&args[0]), Path::new(&args[1])) {
        Ok(()) => {
            println!(
                "reference integration: wire, signed logs, transcript and rejection controls passed; no production anchor claim"
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("reference integration failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_directory_is_refused_before_creating_any_artifact() {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "tessera-integration-existing-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("empty existing directory");
        let error = run(Path::new("absent-test-binary"), &directory)
            .expect_err("existing paths are refused");
        assert_eq!(
            error
                .downcast_ref::<io::Error>()
                .expect("filesystem error")
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(
            fs::read_dir(&directory)
                .expect("unchanged directory")
                .count(),
            0
        );
        fs::remove_dir(directory).expect("remove empty scratch directory");
    }
}
